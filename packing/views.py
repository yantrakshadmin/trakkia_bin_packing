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
from matplotlib import pyplot as plt
from packing.utils import calculate_dummy_height, calculate_matrix_details, calculate_volume_used_percentage, convert_to_kg, convert_to_mm, plot_items_in_box_version1, get_box_dimensions

class PackingAPIView(APIView):
    def post(self, request):
        try:
            data = request.data

            required_fields = [
                "L_item", "B_item", "H_item", "dimension_unit",
                "weight_per_item", "weight_unit",
                "margin", "orientations", "box_key"
            ]

            #padding = 0

            missing = [field for field in required_fields if field not in data]
            if missing:
                return Response({"error": f"Missing fields: {', '.join(missing)}"}, status=status.HTTP_400_BAD_REQUEST)

            L_item = convert_to_mm(data["L_item"], data["dimension_unit"])
            B_item = convert_to_mm(data["B_item"], data["dimension_unit"])
            H_item = convert_to_mm(data["H_item"], data["dimension_unit"])
            weight = convert_to_kg(data["weight_per_item"], data["weight_unit"])

            box = get_box_dimensions(data["box_key"])
            if not box:
                return Response({"error": f"Invalid box_key: {data['box_key']}"}, status=status.HTTP_400_BAD_REQUEST)

            main_image, insert_images, insert_config = plot_items_in_box_version1(
                L_box=box["L_box"],
                B_box=box["B_box"],
                H_box=box["H_box"],
                L_item=L_item,
                B_item=B_item,
                H_item=H_item,
                weight_per_item=weight,
                margin=data["margin"],
                user_orientations=data["orientations"],
                max_weight=box.get("max_weight")
            )

            total_inserts = len(insert_config)
            best_orientations = [insert[4] for insert in insert_config]
            best_orientation = best_orientations[0]

            matrix_details_dict = {}
            for orientation in best_orientations:
                matrix, remaining_length, remaining_width = calculate_matrix_details(
                    L_insert=box["L_box"], B_insert=box["B_box"],
                    part_length=L_item, part_width=B_item, part_height=H_item,
                    padding=0, margin=data["margin"], orientation=orientation
                )
                matrix_details_dict[orientation] = matrix
                matrix_detail_str = ", ".join(f"{k}={v}" for k, v in matrix_details_dict.items())

            total_items = sum(insert[5] for insert in insert_config)
            total_weight = total_items * weight

            volume_used = calculate_volume_used_percentage(
                                    data["L_item"], data["B_item"], data["H_item"], 
                                    L_item, B_item, H_item, 
                                    0, data["margin"], 
                                    total_items,
                                    orientation=best_orientation
                                )

            dummy_height = calculate_dummy_height(data["H_item"], L_item, B_item, H_item, data["margin"], 0, best_orientation, total_inserts)
            dummy_space = f"{remaining_length}x{remaining_width}x{dummy_height}"

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
                "total_weight": total_weight
            })

        except Exception as e:
            import traceback
            return Response(
                {"error": str(e), "trace": traceback.format_exc()},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

