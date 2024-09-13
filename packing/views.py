import base64
from io import BytesIO

from django.shortcuts import render
from matplotlib import pyplot as plt
from packing.forms import ItemForm
from packing.plot_algo import get_box_dimensions, plot_items_in_box
from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage


def pack_items_view(request):
    if request.method == 'POST':
        form = ItemForm(request.POST)
        if form.is_valid():
            L_item = form.cleaned_data['length']
            B_item = form.cleaned_data['breadth']
            H_item = form.cleaned_data['height']
            box_key = form.cleaned_data['truck_type']
            padding = form.cleaned_data['padding']
            weight_per_item = form.cleaned_data['weight']
            user_orientations = form.cleaned_data['orientations']

            box_dimensions = get_box_dimensions(box_key)
            L_box = box_dimensions['L_box']
            B_box = box_dimensions['B_box']
            H_box = box_dimensions['H_box']
            max_weight = box_dimensions['max_weight']

            image_path, total_items, total_weight, total_volume, orientation_count = plot_items_in_box(L_box, B_box, H_box, L_item, B_item, H_item, weight_per_item=weight_per_item, padding=padding, user_orientations=user_orientations, max_weight=max_weight)

            image_file = ContentFile(base64.b64decode(image_path), name=f'{box_key}_plot.png')
            image_path = default_storage.save(f'plots/{box_key}_plot.png', image_file)
            image_url = f"{settings.MEDIA_URL}{image_path}"
            
            context = {
                'total_items': total_items,
                'orientation_count': orientation_count,
                'total_weight': round(total_weight, 2),
                'total_volume': round(total_volume/1000, 2),
                'image_path': image_url,
            }
            return render(request, 'packing/result.html', context)
    else:
        form = ItemForm()

    return render(request, 'packing/packing_form.html', {'form': form})