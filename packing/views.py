from django.shortcuts import render
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.conf import settings
import base64
import time
import json

from matplotlib import pyplot as plt

from .forms import SingleItemForm, MultipleItemsForm
from .plot_algo import Item, Box, find_best_box, plot_items_in_box, get_box_dimensions, visualize_packing

def pack_items_view(request):
    context = {}

    if request.method == 'POST':
        item_type = request.POST.get('item_type', 'single')

        if item_type == 'single':
            form = SingleItemForm(request.POST)
            if form.is_valid():
                L_item = form.cleaned_data['length']
                B_item = form.cleaned_data['breadth']
                H_item = form.cleaned_data['height']
                box_key = form.cleaned_data['truck_type']
                padding = 0
                weight_per_item = form.cleaned_data['weight']
                user_orientations = form.cleaned_data['orientations']

                box_dimensions = get_box_dimensions(box_key)
                L_box = box_dimensions['L_box']
                B_box = box_dimensions['B_box']
                H_box = box_dimensions['H_box']
                max_weight = box_dimensions['max_weight']

                image_path, total_items, total_weight, total_volume, orientation_count = plot_items_in_box(
                    L_box, B_box, H_box, L_item, B_item, H_item,
                    weight_per_item=weight_per_item,
                    padding=padding,
                    user_orientations=user_orientations,
                    max_weight=max_weight
                )

                timestamp = int(time.time())
                image_file = ContentFile(base64.b64decode(image_path), name=f'{box_key}_plot.png')
                image_path = default_storage.save(f'plots/{box_key}_plot.png', image_file)
                image_url = f"{settings.MEDIA_URL}{image_path}"

                request.session['image_to_delete'] = image_path
                request.session['image_created_at'] = timestamp

                context.update({
                    'total_items': total_items,
                    'orientation_count': orientation_count,
                    'total_weight': round(total_weight, 2),
                    'total_volume': round(total_volume / 1000, 2),
                    'image_path': image_url,
                })

                if 'image_to_delete' in request.session and 'image_created_at' in request.session:
                    current_time = int(time.time())
                    image_created_at = request.session['image_created_at']
                    if current_time - image_created_at > 300:
                        image_path_to_delete = request.session['image_to_delete']
                        if default_storage.exists(image_path_to_delete):
                            default_storage.delete(image_path_to_delete)
                            del request.session['image_to_delete']
                            del request.session['image_created_at']            
                
            return render(request, 'packing/packing_form.html', context)

        elif item_type == 'multiple':
            form = MultipleItemsForm(request.POST)
            if form.is_valid():
                items_data = request.POST.get('items_data')
                if items_data:
                    items = json.loads(items_data)
                    item_objects = [Item(
                        length=item['length'],
                        width=item['breadth'],
                        height=item['height'],
                        quantity=item['quantity'],
                        weight=item['weight']
                    ) for item in items]

                    available_boxes = {
                        "Tempo_407": {"length": 2896, "width": 1676, "height": 1676, "max_weight": 2500},
                        "13_Feet": {"length": 3962, "width": 1676, "height": 2134, "max_weight": 3500},
                        "14_Feet": {"length": 4267, "width": 1829, "height": 1829, "max_weight": 4000},
                        "17_Feet": {"length": 5182, "width": 1829, "height": 2134, "max_weight": 6000},
                        "20_ft_sxl": {"length": 6096, "width": 2438, "height": 2438, "max_weight": 7000},
                        "24_ft_sxl": {"length": 7315, "width": 2438, "height": 2438, "max_weight": 7000},
                        "32_ft_sxl": {"length": 9754, "width": 2438, "height": 2438, "max_weight": 7000},
                        "32_ft_sxl_HQ": {"length": 9754, "width": 2743, "height": 2896, "max_weight": 7000},
                        "32_ft_mxl": {"length": 9754, "width": 2438, "height": 2438, "max_weight": 15000},
                        "32_ft_mxl_HQ": {"length": 9754, "width": 2743, "height": 2896, "max_weight": 14000},
                    }
                    best_box_key, best_fit_score, best_positions, best_orientations = find_best_box(item_objects, available_boxes)

                    if best_box_key:
                        best_box = Box(**available_boxes[best_box_key])
                        print(f"Best box: {best_box_key} with {best_fit_score} leftover volume")
                        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
                        visualize_packing(best_box, item_objects, best_positions, best_orientations)
                        plt.close(fig)

                        image_path = f'{best_box_key}_plot.png'
                        image_file = ContentFile(fig.savefig(image_path, format='png'))
                        image_url = default_storage.save(f'plots/{image_path}', image_file)
                        image_url = f"{settings.MEDIA_URL}{image_url}"

                        context.update({
                            'best_box': best_box_key,
                            'image_path': image_url,
                        })

                return render(request, 'packing/packing_form.html', context)

    else:
        form = SingleItemForm()

    context['form'] = form
    return render(request, 'packing/packing_form.html', context)

