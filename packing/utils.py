import base64
from io import BytesIO
import plotly.graph_objects as go
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import requests
import itertools

BOX_DIMENSIONS = {
    "Pickup": {"L_box": 2740, "B_box": 1676, "H_box": 1828, "max_weight": 1000},
    "Tempo_407": {"L_box": 2896, "B_box": 1676, "H_box": 1676, "max_weight": 2500},
    "13_Feet": {"L_box": 3962, "B_box": 1676, "H_box": 2134, "max_weight": 3500},
    "14_Feet": {"L_box": 4267, "B_box": 1829, "H_box": 1829, "max_weight": 4000},
    "17_Feet": {"L_box": 5182, "B_box": 1829, "H_box": 2134, "max_weight": 6000},
    "20_ft_sxl": {"L_box": 6096, "B_box": 2438, "H_box": 2438, "max_weight": 7000},
    "24_ft_sxl": {"L_box": 7315, "B_box": 2438, "H_box": 2438, "max_weight": 7000},
    "32_ft_sxl": {"L_box": 9754, "B_box": 2438, "H_box": 2438, "max_weight": 8000},
    "32_ft_sxl_HQ": {"L_box": 9754, "B_box": 2529, "H_box": 3048, "max_weight": 8000},
    "32_ft_mxl": {"L_box": 9754, "B_box": 2438, "H_box": 2438, "max_weight": 16000},
    "32_ft_mxl_HQ": {"L_box": 9754, "B_box": 2529, "H_box": 3048, "max_weight": 16000},
    "PLS12801": {"L_box": 1150, "B_box": 750, "H_box": 790, "max_weight": 600},
    "PLS12802": {"L_box": 1150, "B_box": 750, "H_box": 490, "max_weight": 600},
    "FLC": {"L_box": 1100, "B_box": 900, "H_box": 760, "max_weight": 800},
    "CRT6412": {"L_box": 550, "B_box": 360, "H_box": 110, "max_weight": 14},
    "CRT6418": {"L_box": 550, "B_box": 360, "H_box": 170, "max_weight": 14},
    "CRT6423": {"L_box": 550, "B_box": 360, "H_box": 225, "max_weight": 14},
    "CRT6434": {"L_box": 550, "B_box": 360, "H_box": 340, "max_weight": 14},
    "CRT4312": {"L_box": 350, "B_box": 260, "H_box": 110, "max_weight": 14},
    "CRT4323": {"L_box": 350, "B_box": 260, "H_box": 225, "max_weight": 14},
    "CRT6435": {"L_box": 550, "B_box": 360, "H_box": 340, "max_weight": 14}
}

def get_box_dimensions(box_key):
    if box_key.startswith("Palletized "):
        box_key = box_key.replace("Palletized ", "")
    return BOX_DIMENSIONS.get(box_key)

def calculate_pocket_dimensions(L_item, B_item, H_item, padding, orientation):
    dimensions = {
        'L_B_H': (L_item, B_item, H_item),
        'L_H_B': (L_item, H_item, B_item),
        'B_L_H': (B_item, L_item, H_item),
        'B_H_L': (B_item, H_item, L_item),
        'H_L_B': (H_item, L_item, B_item),
        'H_B_L': (H_item, B_item, L_item)
    }
    dim = dimensions[orientation]
    return (
        dim[0] + padding,
        dim[1] + padding,
        dim[2] + padding
    )


def convert_to_mm(value, unit):
    unit = unit.lower()
    if unit == "cm":
        return value * 10
    elif unit == "m":
        return value * 1000
    elif unit == "inch":
        return value * 25.4
    elif unit == "feet":
        return value * 304.8
    elif unit == "mm":
        return value
    else:
        raise ValueError(f"Unsupported length unit: {unit}")

def convert_to_kg(value, unit):
    unit = unit.lower()
    if unit == "g":
        return value / 1000
    elif unit == "tonne":
        return value * 1000
    elif unit == "kg":
        return value
    else:
        raise ValueError(f"Unsupported weight unit: {unit}")
    

# def plot_items_in_box_version1(L_box, B_box, H_box, L_item, B_item, H_item, weight_per_item, margin, user_orientations, max_weight=None):
#     def max_items_with_inserts():
#         L_insert = L_box
#         B_insert = B_box

#         remaining_height = H_box
#         insert_index = 0
#         current_weight = 0
#         insert_config = []

#         while remaining_height > 0:
#             best_local = None
#             best_items = 0
#             best_height = 0

#             for orientation in user_orientations:
#                 dims = {
#                     'L_B_H': (L_item, B_item, H_item),
#                     'L_H_B': (L_item, H_item, B_item),
#                     'B_L_H': (B_item, L_item, H_item),
#                     'B_H_L': (B_item, H_item, L_item),
#                     'H_L_B': (H_item, L_item, B_item),
#                     'H_B_L': (H_item, B_item, L_item)
#                 }
#                 dim = dims[orientation]
#                 x, y, z = dim[0] + margin, dim[1] + margin, dim[2] + margin

#                 num_L = int(L_insert // x)
#                 num_B = int(B_insert // y)
#                 num_H = int(remaining_height // z)

#                 total_items = num_L * num_B * min(num_H, 1)

#                 if max_weight and weight_per_item:
#                     if (current_weight + total_items * weight_per_item) > max_weight:
#                         total_items = int((max_weight - current_weight) // weight_per_item)

#                 if total_items > best_items:
#                     best_items = total_items
#                     best_local = (insert_index, L_insert, B_insert, z, orientation, total_items)
#                     best_height = z

#             if best_local:
#                 insert_config.append(best_local)
#                 current_weight += best_items * weight_per_item
#                 remaining_height -= best_height
#                 insert_index += 1
#             else:
#                 break

#         return insert_config

#     def plot_items_in_insert(ax, x_start, y_start, z_start, L_insert, B_insert, H_insert, orientation, max_items):
#         dims = {
#             'L_B_H': (L_item, B_item, H_item),
#             'L_H_B': (L_item, H_item, B_item),
#             'B_L_H': (B_item, L_item, H_item),
#             'B_H_L': (B_item, H_item, L_item),
#             'H_L_B': (H_item, L_item, B_item),
#             'H_B_L': (H_item, B_item, L_item)
#         }
#         dim = dims[orientation]
#         x_dim, y_dim, z_dim = dim[0] + margin, dim[1] + margin, dim[2] + margin

#         max_L = int(L_insert // x_dim)
#         max_B = int(B_insert // y_dim)

#         num_B = min(max_B, max_items)
#         num_L = min(max_L, (max_items + num_B - 1) // num_B)

#         used_L = num_L * x_dim
#         used_B = num_B * y_dim

#         x_offset = (L_insert - used_L) / 2
#         y_offset = (B_insert - used_B) / 2

#         count = 0
#         for i in range(num_L):
#             for j in range(num_B):
#                 if count >= max_items:
#                     return
#                 x = x_start + x_offset + i * x_dim
#                 y = y_start + y_offset + j * y_dim
#                 z = z_start

#                 item_coords = np.array([
#                     [x, y, z],
#                     [x + dim[0], y, z],
#                     [x + dim[0], y + dim[1], z],
#                     [x, y + dim[1], z],
#                     [x, y, z + dim[2]],
#                     [x + dim[0], y, z + dim[2]],
#                     [x + dim[0], y + dim[1], z + dim[2]],
#                     [x, y + dim[1], z + dim[2]]
#                 ])

#                 edges = [
#                     [item_coords[0], item_coords[1], item_coords[2], item_coords[3]],
#                     [item_coords[4], item_coords[5], item_coords[6], item_coords[7]],
#                     [item_coords[0], item_coords[1], item_coords[5], item_coords[4]],
#                     [item_coords[2], item_coords[3], item_coords[7], item_coords[6]],
#                     [item_coords[1], item_coords[2], item_coords[6], item_coords[5]],
#                     [item_coords[4], item_coords[7], item_coords[3], item_coords[0]]
#                 ]

#                 color_map = {
#                     'L_B_H': 'blue', 'L_H_B': 'lightblue',
#                     'B_L_H': 'grey', 'B_H_L': 'lightgreen',
#                     'H_L_B': 'red', 'H_B_L': 'salmon'
#                 }

#                 item = Poly3DCollection(edges, alpha=0.6, facecolors=color_map.get(orientation, 'cyan'), linewidths=0.5, edgecolors='k')
#                 ax.add_collection3d(item)
#                 count += 1

#     insert_config = max_items_with_inserts()

#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     ax.set_xlim([0, L_box])
#     ax.set_ylim([0, B_box])
#     ax.set_zlim([0, H_box])
#     ax.set_xlabel('Length (mm)')
#     ax.set_ylabel('Breadth (mm)')
#     ax.set_zlabel('Height (mm)')
#     ax.set_box_aspect([L_box, B_box, H_box])

#     r = [[0, L_box], [0, B_box], [0, H_box]]
#     verts = [[r[0][0], r[1][0], r[2][0]], [r[0][1], r[1][0], r[2][0]], [r[0][1], r[1][1], r[2][0]], [r[0][0], r[1][1], r[2][0]],
#              [r[0][0], r[1][0], r[2][1]], [r[0][1], r[1][0], r[2][1]], [r[0][1], r[1][1], r[2][1]], [r[0][0], r[1][1], r[2][1]]]
#     faces = [[verts[0], verts[1], verts[2], verts[3]], [verts[4], verts[5], verts[6], verts[7]], [verts[0], verts[1], verts[5], verts[4]],
#              [verts[2], verts[3], verts[7], verts[6]], [verts[1], verts[2], verts[6], verts[5]], [verts[4], verts[7], verts[3], verts[0]]]
#     ax.add_collection3d(Poly3DCollection(faces, alpha=0.1, facecolors='white', linewidths=0.5, edgecolors='k'))

#     z_level = margin
#     for insert in insert_config:
#         _, L_insert, B_insert, H_insert, orientation, max_items = insert

#         plot_items_in_insert(ax, 0, 0, z_level, L_insert, B_insert, H_insert, orientation, max_items)
#         z_level += H_insert

#     buffer = BytesIO()
#     plt.savefig(buffer, format='png', bbox_inches='tight')
#     buffer.seek(0)
#     img_str = base64.b64encode(buffer.read()).decode('utf-8')
#     plt.close()

#     insert_images = []
#     if insert_config:
#         unique_orientations = set(ins[4] for ins in insert_config)

#         for orientation in unique_orientations:
#             for ins in insert_config:
#                 if ins[4] == orientation:
#                     _, L_insert, B_insert, H_insert, _, max_items = ins
#                     break

#             fig2 = plt.figure(figsize=(10, 8))
#             ax2 = fig2.add_subplot(111, projection='3d')
#             ax2.set_title(f"Insert View - Orientation {orientation}")
#             ax2.set_xlim([0, L_insert])
#             ax2.set_ylim([0, B_insert])
#             ax2.set_zlim([0, H_insert])
#             ax2.set_box_aspect([L_insert, B_insert, H_insert])

#             plot_items_in_insert(ax2, 0, 0, 0, L_insert, B_insert, H_insert, orientation, max_items)
#             buffer2 = BytesIO()
#             plt.savefig(buffer2, format='png', bbox_inches='tight')
#             buffer2.seek(0)
#             insert_img_str = base64.b64encode(buffer2.read()).decode('utf-8')
#             plt.close()
#             insert_images.append((orientation,  insert_img_str))
#     else:
#         insert_images = []
    
#     return img_str, insert_images, insert_config


def plot_items_in_box_version1(L_box, B_box, H_box, L_item, B_item, H_item, weight_per_item, margin, user_orientations, max_weight=None):
    def max_items_with_inserts():
        L_insert = L_box
        B_insert = B_box

        remaining_height = H_box
        insert_index = 0
        current_weight = 0
        insert_config = []

        while remaining_height > 0:
            best_local = None
            best_items = 0
            best_height = 0

            for orientation in user_orientations:
                dims = {
                    'L_B_H': (L_item, B_item, H_item),
                    'L_H_B': (L_item, H_item, B_item),
                    'B_L_H': (B_item, L_item, H_item),
                    'B_H_L': (B_item, H_item, L_item),
                    'H_L_B': (H_item, L_item, B_item),
                    'H_B_L': (H_item, B_item, L_item)
                }
                dim = dims[orientation]
                x, y, z = dim[0] + margin, dim[1] + margin, dim[2] + margin

                num_L = int(L_insert // x)
                num_B = int(B_insert // y)
                num_H = int(remaining_height // z)

                total_items = num_L * num_B * min(num_H, 1)

                if max_weight and weight_per_item:
                    if (current_weight + total_items * weight_per_item) > max_weight:
                        total_items = int((max_weight - current_weight) // weight_per_item)

                if total_items > best_items:
                    best_items = total_items
                    best_local = (insert_index, L_insert, B_insert, z, orientation, total_items)
                    best_height = z

            if best_local:
                insert_config.append(best_local)
                current_weight += best_items * weight_per_item
                remaining_height -= best_height
                insert_index += 1
            else:
                break

        return insert_config

    def plot_items_in_insert(data_list, x_start, y_start, z_start, L_insert, B_insert, H_insert, orientation, max_items):
        dims = {
            'L_B_H': (L_item, B_item, H_item),
            'L_H_B': (L_item, H_item, B_item),
            'B_L_H': (B_item, L_item, H_item),
            'B_H_L': (B_item, H_item, L_item),
            'H_L_B': (H_item, L_item, B_item),
            'H_B_L': (H_item, B_item, L_item)
        }
        dim = dims[orientation]
        x_dim, y_dim, z_dim = dim[0] + margin, dim[1] + margin, dim[2] + margin

        max_L = int(L_insert // x_dim)
        max_B = int(B_insert // y_dim)

        num_B = min(max_B, max_items)
        num_L = min(max_L, (max_items + num_B - 1) // num_B)

        used_L = num_L * x_dim
        used_B = num_B * y_dim

        x_offset = (L_insert - used_L) / 2
        y_offset = (B_insert - used_B) / 2

        color_map = {
            'L_B_H': 'blue', 'L_H_B': 'lightblue',
            'B_L_H': 'grey', 'B_H_L': 'lightgreen',
            'H_L_B': 'red', 'H_B_L': 'salmon'
        }

        count = 0
        for i in range(num_L):
            for j in range(num_B):
                if count >= max_items:
                    return
                x = x_start + x_offset + i * x_dim
                y = y_start + y_offset + j * y_dim
                z = z_start

                dx, dy, dz = dim[0], dim[1], dim[2]

                cube = go.Mesh3d(
                    x=[x, x + dx, x + dx, x, x, x + dx, x + dx, x],
                    y=[y, y, y + dy, y + dy, y, y, y + dy, y + dy],
                    z=[z, z, z, z, z + dz, z + dz, z + dz, z + dz],
                    i=[0, 0, 0, 1, 1, 2, 4, 5, 6, 4, 5, 1],
                    j=[1, 2, 3, 2, 5, 3, 5, 6, 7, 0, 4, 5],
                    k=[2, 3, 0, 5, 6, 0, 6, 7, 4, 4, 0, 1],
                    color=color_map.get(orientation, 'cyan'),
                    opacity=0.6,
                    showscale=False
                )
                data_list.append(cube)
                count += 1

    insert_config = max_items_with_inserts()

    fig_data = []

    fig_data.append(go.Mesh3d(
        x=[0, L_box, L_box, 0, 0, L_box, L_box, 0],
        y=[0, 0, B_box, B_box, 0, 0, B_box, B_box],
        z=[0, 0, 0, 0, H_box, H_box, H_box, H_box],
        i=[0, 0, 0, 1, 1, 2, 4, 5, 6, 4, 5, 1],
        j=[1, 2, 3, 2, 5, 3, 5, 6, 7, 0, 4, 5],
        k=[2, 3, 0, 5, 6, 0, 6, 7, 4, 4, 0, 1],
        color='white',
        opacity=0.1,
        showscale=False
    ))

    z_level = margin
    for insert in insert_config:
        _, L_insert, B_insert, H_insert, orientation, max_items = insert
        plot_items_in_insert(fig_data, 0, 0, z_level, L_insert, B_insert, H_insert, orientation, max_items)
        z_level += H_insert

    fig = go.Figure(data=fig_data)
    fig.update_layout(
            scene=dict(
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                aspectmode='data'
            ),
            margin=dict(l=0, r=0, b=0, t=0)
        )
    
    insert_figs = []
    unique_orientations = set(ins[4] for ins in insert_config)

    for orientation in unique_orientations:
        for ins in insert_config:
            if ins[4] == orientation:
                _, L_insert, B_insert, H_insert, _, max_items = ins
                break

        insert_data = []
        plot_items_in_insert(insert_data, 0, 0, 0, L_insert, B_insert, H_insert, orientation, max_items)

        insert_fig = go.Figure(data=insert_data)
        insert_fig.update_layout(
            title=f"Insert View - Orientation {orientation}",
            scene=dict(
                xaxis_title='Length',
                yaxis_title='Breadth',
                zaxis_title='Height',
                xaxis=dict(nticks=10, range=[0, L_insert]),
                yaxis=dict(nticks=10, range=[0, B_insert]),
                zaxis=dict(nticks=10, range=[0, H_insert]),
                aspectratio=dict(x=L_insert, y=B_insert, z=H_insert)
            ),
            margin=dict(l=0, r=0, t=50, b=0)
        )
        insert_figs.append((orientation, insert_fig))

    return fig, insert_figs, insert_config


logger = logging.getLogger(__name__)

def get_base64_image_from_url(url):
    try:
        logger.info(f'Fetching image from URL: {url}')
        response = requests.get(url)
        if response.status_code == 200:
            image_data = response.content
            base64_encoded_image = base64.b64encode(image_data).decode('utf-8')
            logger.info('Image successfully fetched and encoded to base64')
            return f'data:{response.headers["Content-Type"]};base64,{base64_encoded_image}'
        else:
            logger.error(f'Error fetching image: {response.status_code}')
            return None
    except Exception as e:
        logger.error(f'Error fetching image from URL: {e}')
        return None
    

def calculate_matrix_details(L_insert, B_insert, part_length, part_width, part_height, padding, margin, orientation):
    dimensions = {
        'L_B_H': (part_length, part_width),
        'L_H_B': (part_length, part_height),
        'B_L_H': (part_width, part_length),
        'B_H_L': (part_width, part_height),
        'H_L_B': (part_height, part_length),
        'H_B_L': (part_height, part_width)
    }
    
    adjusted_length, adjusted_width = dimensions[orientation]
    
    adjusted_length += margin
    adjusted_width += margin

    num_rows = 0
    used_length = padding

    while used_length + adjusted_length + padding <= L_insert:
        used_length += adjusted_length + padding
        num_rows += 1

    num_columns = 0
    used_width = padding

    while used_width + adjusted_width + padding <= B_insert:
        used_width += adjusted_width + padding
        num_columns += 1

    remaining_length = L_insert - (num_rows * adjusted_length + (num_rows - 1) * padding)
    remaining_width = B_insert - (num_columns * adjusted_width + (num_columns - 1) * padding)

    matrix_details = f"{num_rows}x{num_columns}"
    return matrix_details, remaining_length, remaining_width


def calculate_volume_used_percentage(L_box, B_box, H_box, L_item, B_item, H_item, padding, margin, total_items, orientation):
    dimensions = {
            'L_B_H': (L_item, B_item, H_item),
            'L_H_B': (L_item, H_item, B_item),
            'B_L_H': (B_item, L_item, H_item),
            'B_H_L': (B_item, H_item, L_item),
            'H_L_B': (H_item, L_item, B_item),
            'H_B_L': (H_item, B_item, L_item)
        }
    
    length, width, height= dimensions[orientation]
    
    box_volume = L_box * B_box * H_box

    adjusted_item_volume = (
        (length + padding + margin)
        * (width + padding + margin)
        * (height + padding + margin)
    )
    total_item_volume = adjusted_item_volume * total_items

    if box_volume > 0:
        volume_used_percentage = (total_item_volume / box_volume) * 100
    else:
        volume_used_percentage = 0

    return round(volume_used_percentage, 2), total_item_volume


def calculate_dummy_height(H_box, L_item, B_item, H_item, margin, padding, best_orientation, total_inserts):
    dimensions = {
            'L_B_H': (L_item, B_item, H_item),
            'L_H_B': (L_item, H_item, B_item),
            'B_L_H': (B_item, L_item, H_item),
            'B_H_L': (B_item, H_item, L_item),
            'H_L_B': (H_item, L_item, B_item),
            'H_B_L': (H_item, B_item, L_item)
        }    
    part_height = dimensions[best_orientation][2]
    insert_height = part_height + margin + padding 
    
    total_height_used = insert_height * total_inserts
    
    dummy_height = H_box - total_height_used
    
    return dummy_height


def plotly_figure_to_imgur_url(fig, imgur_client_id):
    # Save Plotly figure to PNG
    fig.write_image("temp_plot.png", width=1000, height=800)

    # Encode image to base64
    with open("temp_plot.png", "rb") as f:
        b64_image = base64.b64encode(f.read()).decode("utf-8")

    # Upload to Imgur
    headers = {'Authorization': f'Client-ID {imgur_client_id}'}
    data = {'image': b64_image, 'type': 'base64'}
    response = requests.post("https://api.imgur.com/3/image", headers=headers, data=data)

    if response.status_code == 200:
        return response.json()['data']['link']  # ✅ This is your image URL
    else:
        raise Exception("Failed to upload image: " + response.text)