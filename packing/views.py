from django.shortcuts import render
from rest_framework import status
from django.utils.timezone import now
from rest_framework.response import Response
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.conf import settings
import base64
import time
import json
from django.http import HttpResponse
from rest_framework.views import APIView
import uuid
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from packing.utils import calculate_dummy_height, calculate_matrix_details, calculate_volume_used_percentage, convert_to_kg, convert_to_mm, plot_items_in_box_version1, get_box_dimensions

def home(request):
    return render(request, 'packing/home.html', {'data': 'Hello, world!'})

class PackingAPIView(APIView):
    def post(self, request):
        try:
            data = request.data

            required_fields = [
                "L_item", "B_item", "H_item", "dimension_unit",
                "weight_per_item", "weight_unit",
                "orientations", "box_key"
            ]

            #padding = 0

            missing = [field for field in required_fields if field not in data]
            if missing:
                return Response({"error": f"Missing fields: {', '.join(missing)}"}, status=status.HTTP_400_BAD_REQUEST)

            L_item = convert_to_mm(data["L_item"], data["length_unit"])
            B_item = convert_to_mm(data["B_item"], data["length_unit"])
            H_item = convert_to_mm(data["H_item"], data["length_unit"])
            weight = convert_to_kg(data["weight_per_item"], data["weight_unit"])
            raw_margin = data["margin"]
            margin_value = 0 if raw_margin is None else raw_margin
            margin = convert_to_mm(margin_value, data["length_unit"])

            if data["box_key"] == "Others":
                try:
                    L_box = convert_to_mm(data["L_box"], data["dimension_unit"])
                    B_box = convert_to_mm(data["B_box"], data["dimension_unit"])
                    H_box = convert_to_mm(data["H_box"], data["dimension_unit"])
                    max_weight = convert_to_kg(data["max_weight"], data["max_weight_unit"])
                except KeyError as e:
                    return Response({"error": f"Missing dimension in payload: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
                
            else:
                box = get_box_dimensions(data["box_key"])
                if not box:
                    return Response({"error": f"Invalid box_key: {data['box_key']}"}, status=status.HTTP_400_BAD_REQUEST)
                L_box = box["L_box"]
                B_box = box["B_box"]
                H_box = box["H_box"]
                max_weight = box.get("max_weight")

            main_image, insert_images, insert_config = plot_items_in_box_version1(
                L_box=L_box,
                B_box=B_box,
                H_box=H_box,
                L_item=L_item,
                B_item=B_item,
                H_item=H_item,
                weight_per_item=weight,
                margin=margin,
                user_orientations=data["orientations"],
                max_weight=max_weight
            )

            print(insert_config, "insert config")


            total_inserts = len(insert_config)
            best_orientations = [insert[4] for insert in insert_config]
            best_orientation = best_orientations[0]

            matrix_details_dict = {}
            for orientation in best_orientations:
                matrix, remaining_length, remaining_width = calculate_matrix_details(
                    L_insert=L_box, B_insert=B_box,
                    part_length=L_item, part_width=B_item, part_height=H_item,
                    padding=0, margin=margin, orientation=orientation
                )
                matrix_details_dict[orientation] = matrix
                matrix_detail_str = ", ".join(f"{k}={v}" for k, v in matrix_details_dict.items())

            total_items = sum(insert[5] for insert in insert_config)
            total_weight = total_items * weight

            volume_used, total_volume = calculate_volume_used_percentage(
                                    L_box, B_box, H_box, 
                                    L_item, B_item, H_item, 
                                    0, margin, 
                                    total_items,
                                    orientation=best_orientation
                                )

            dummy_height = calculate_dummy_height(data["H_item"], L_item, B_item, H_item, margin, 0, best_orientation, total_inserts)
            dummy_space = f"{remaining_length}x{remaining_width}x{dummy_height}"
            # items_volume = L_item * B_item * H_item
            loaded_volume = L_box * B_box * H_box
            loaded_weight_percentage = (total_weight/max_weight)*100

            # unique_id = uuid.uuid4().hex
            # timestamp = now().strftime('%Y%m%d%H%M%S')
            # file_name = f'{timestamp}_{unique_id}.png'
            # image_file = ContentFile(base64.b64decode(main_image), name=file_name)
            # image_path = default_storage.save(f'plots/{file_name}', image_file)
            # image_url = request.build_absolute_uri(f"{settings.MEDIA_URL}{image_path}")
            image_data_uri = f"data:image/png;base64,{main_image}"

            return Response({
                "main_image": image_data_uri,
                "insert_config": [
                    {
                        "insert_index": cfg[0],
                        "L_insert": cfg[1],
                        "B_insert": cfg[2],
                        "H_insert": cfg[3],
                        "orientation": cfg[4],
                        "items_in_insert": cfg[5]
                    } for cfg in insert_config
                ],
                "dummy_space": dummy_space,
                "volumetric_weight": volume_used,
                "matrix_details": matrix_detail_str,
                "total_weight": total_weight,
                "items_volume": round(total_volume/1000000000, 2),
                "loaded_volume": round(loaded_volume/1000000000, 2),
                "loaded_weight": max_weight,
                "total_items": total_items,
                "solution": data["box_key"],
                "loaded_weight_percentage": round(loaded_weight_percentage, 2)
            })

        except Exception as e:
            import traceback
            return Response(
                {"error": str(e), "trace": traceback.format_exc()},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

