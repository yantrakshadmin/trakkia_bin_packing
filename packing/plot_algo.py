import base64
from io import BytesIO
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

def plot_items_in_box(L_box, B_box, H_box, L_item, B_item, H_item, weight_per_item, padding, user_orientations, max_weight=None):
    def can_place_item(x_start, y_start, z_start, dim, occupied_spaces):
        x_end, y_end, z_end = x_start + dim[0], y_start + dim[1], z_start + dim[2]

        if x_end > L_box or y_end > B_box or z_end > H_box:
            return False

        for (ox_start, oy_start, oz_start, odim) in occupied_spaces:
            ox_end = ox_start + odim[0]
            oy_end = oy_start + odim[1]
            oz_end = oz_start + odim[2]

            if (x_start < ox_end and x_end > ox_start and
                y_start < oy_end and y_end > oy_start and
                z_start < oz_end and z_end > oz_start):
                return False
        return True

    def items_in_box(L_box, B_box, H_box, L_item, B_item, H_item, padding, orientations, max_weight, weight_per_item):
        dimensions = {
            'L_B_H': (L_item, B_item, H_item),
            'L_H_B': (L_item, H_item, B_item),
            'B_L_H': (B_item, L_item, H_item),
            'B_H_L': (B_item, H_item, L_item),
            'H_L_B': (H_item, L_item, B_item),
            'H_B_L': (H_item, B_item, L_item)
        }

        total_items = 0
        total_volume = 0
        item_positions = []
        occupied_spaces = []
        orientation_count = {orientation: 0 for orientation in orientations} 
        for orientation in orientations:
            dim = dimensions[orientation]
            num_L = int(L_box / (dim[0] + padding))
            num_B = int(B_box / (dim[1] + padding))
            num_H = int(H_box / (dim[2] + padding))

            for i in range(num_L):
                for j in range(num_B):
                    for k in range(num_H):
                        if max_weight and total_items * weight_per_item >= max_weight:
                            return total_items, item_positions, orientation_count

                        x = i * (dim[0] + padding)
                        y = j * (dim[1] + padding)
                        z = k * (dim[2] + padding)

                        if can_place_item(x, y, z, dim, occupied_spaces):
                            item_positions.append((x, y, z, dim, orientation))
                            occupied_spaces.append((x, y, z, dim))
                            total_items += 1
                            orientation_count[orientation] += 1
                            total_volume += dim[0] * dim[1] * dim[2]  # Volume of the item
                        else:
                            continue

        total_weight = total_items * weight_per_item

        return total_items, total_weight, total_volume, item_positions, orientation_count

    total_items, total_weight, total_volume, item_positions, orientation_count = items_in_box(L_box, B_box, H_box, L_item, B_item, H_item, padding, user_orientations, max_weight, weight_per_item)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    def create_box(ax, L_box, B_box, H_box):
        r = [[0, L_box], [0, B_box], [0, H_box]]
        vertices = [
            [r[0][0], r[1][0], r[2][0]],
            [r[0][1], r[1][0], r[2][0]],
            [r[0][1], r[1][1], r[2][0]],
            [r[0][0], r[1][1], r[2][0]],
            [r[0][0], r[1][0], r[2][1]],
            [r[0][1], r[1][0], r[2][1]],
            [r[0][1], r[1][1], r[2][1]],
            [r[0][0], r[1][1], r[2][1]]
        ]

        edges = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[4], vertices[7], vertices[3], vertices[0]]
        ]

        box = Poly3DCollection(edges, alpha=0.1, facecolors='white', linewidths=0.5, edgecolors='k')
        ax.add_collection3d(box)

    def plot_items(ax, item_positions, padding):
        colors = {
            'L_B_H': 'blue',
            'L_H_B': 'lightblue',
            'B_L_H': 'grey',
            'B_H_L': 'lightgreen',
            'H_L_B': 'red',
            'H_B_L': 'salmon'
        }

        for x, y, z, dim, orientation in item_positions:
            item_color = colors[orientation]
            item_coords = np.array([
                [x, y, z],
                [x + dim[0], y, z],
                [x + dim[0], y + dim[1], z],
                [x, y + dim[1], z],
                [x, y, z + dim[2]],
                [x + dim[0], y, z + dim[2]],
                [x + dim[0], y + dim[1], z + dim[2]],
                [x, y + dim[1], z + dim[2]]
            ])

            edges = [
                [item_coords[0], item_coords[1], item_coords[2], item_coords[3]],
                [item_coords[4], item_coords[5], item_coords[6], item_coords[7]],
                [item_coords[0], item_coords[1], item_coords[5], item_coords[4]],
                [item_coords[2], item_coords[3], item_coords[7], item_coords[6]],
                [item_coords[1], item_coords[2], item_coords[6], item_coords[5]],
                [item_coords[4], item_coords[7], item_coords[3], item_coords[0]]
            ]

            item = Poly3DCollection(edges, alpha=0.6, facecolors=item_color, linewidths=0.5, edgecolors='k')
            ax.add_collection3d(item)

    create_box(ax, L_box, B_box, H_box)
    plot_items(ax, item_positions, padding)

    ax.set_xlim(0, L_box)
    ax.set_ylim(0, B_box)
    ax.set_zlim(0, H_box)

    ax.set_xlabel('Length (mm)')
    ax.set_ylabel('Breadth (mm)')
    ax.set_zlabel('Height (mm)')

    ax.set_box_aspect([np.ptp([0, L_box]), np.ptp([0, B_box]), np.ptp([0, H_box])])

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    
    return img_str, total_items, total_weight, total_volume, orientation_count



# def plot_items_in_box(L_box, B_box, H_box, L_item, B_item, H_item, weight_per_item, padding, user_orientations, max_weight=None):
#     def can_place_item(x_start, y_start, z_start, dim, occupied_spaces):
#         x_end, y_end, z_end = x_start + dim[0], y_start + dim[1], z_start + dim[2]

#         if x_end > L_box or y_end > B_box or z_end > H_box:
#             return False

#         for (ox_start, oy_start, oz_start, odim) in occupied_spaces:
#             ox_end = ox_start + odim[0]
#             oy_end = oy_start + odim[1]
#             oz_end = oz_start + odim[2]

#             if (x_start < ox_end and x_end > ox_start and
#                 y_start < oy_end and y_end > oy_start and
#                 z_start < oz_end and z_end > oz_start):
#                 return False
#         return True

#     def items_in_box(L_box, B_box, H_box, L_item, B_item, H_item, padding, orientations, max_weight, weight_per_item):
#         dimensions = {
#             'L_B_H': (L_item, B_item, H_item),
#             'L_H_B': (L_item, H_item, B_item),
#             'B_L_H': (B_item, L_item, H_item),
#             'B_H_L': (B_item, H_item, L_item),
#             'H_L_B': (H_item, L_item, B_item),
#             'H_B_L': (H_item, B_item, L_item)
#         }

#         best_orientation_data = None
#         best_total_items = 0

#         # Loop through each orientation and compute the max number of items that can be placed
#         for orientation in orientations:
#             dim = dimensions[orientation]
#             total_items = 0
#             item_positions = []
#             occupied_spaces = []
#             orientation_count = {orientation: 0}

#             num_L = int(L_box / (dim[0] + padding))
#             num_B = int(B_box / (dim[1] + padding))
#             num_H = int(H_box / (dim[2] + padding))

#             for i in range(num_L):
#                 for j in range(num_B):
#                     for k in range(num_H):
#                         if max_weight and total_items * weight_per_item >= max_weight:
#                             break

#                         x = i * (dim[0] + padding)
#                         y = j * (dim[1] + padding)
#                         z = k * (dim[2] + padding)

#                         if can_place_item(x, y, z, dim, occupied_spaces):
#                             item_positions.append((x, y, z, dim, orientation))
#                             occupied_spaces.append((x, y, z, dim))
#                             total_items += 1
#                             orientation_count[orientation] += 1

#             total_volume = total_items * dim[0] * dim[1] * dim[2]
#             total_weight = total_items * weight_per_item

#             # Track the best orientation in terms of total items placed
#             if total_items > best_total_items:
#                 best_total_items = total_items
#                 best_orientation_data = {
#                     'orientation': orientation,
#                     'total_items': total_items,
#                     'total_weight': total_weight,
#                     'total_volume': total_volume,
#                     'item_positions': item_positions,
#                     'orientation_count': orientation_count
#                 }

#         return best_orientation_data

#     # Get the best orientation based on item placement
#     best_orientation_data = items_in_box(L_box, B_box, H_box, L_item, B_item, H_item, padding, user_orientations, max_weight, weight_per_item)

#     if not best_orientation_data:
#         return "No items could be placed", 0, 0, 0, {}

#     item_positions = best_orientation_data['item_positions']
#     total_items = best_orientation_data['total_items']
#     total_weight = best_orientation_data['total_weight']
#     total_volume = best_orientation_data['total_volume']
#     orientation_count = best_orientation_data['orientation_count']

#     # Plotting
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     def create_box(ax, L_box, B_box, H_box):
#         r = [[0, L_box], [0, B_box], [0, H_box]]
#         vertices = [
#             [r[0][0], r[1][0], r[2][0]],
#             [r[0][1], r[1][0], r[2][0]],
#             [r[0][1], r[1][1], r[2][0]],
#             [r[0][0], r[1][1], r[2][0]],
#             [r[0][0], r[1][0], r[2][1]],
#             [r[0][1], r[1][0], r[2][1]],
#             [r[0][1], r[1][1], r[2][1]],
#             [r[0][0], r[1][1], r[2][1]]
#         ]

#         edges = [
#             [vertices[0], vertices[1], vertices[2], vertices[3]],
#             [vertices[4], vertices[5], vertices[6], vertices[7]],
#             [vertices[0], vertices[1], vertices[5], vertices[4]],
#             [vertices[2], vertices[3], vertices[7], vertices[6]],
#             [vertices[1], vertices[2], vertices[6], vertices[5]],
#             [vertices[4], vertices[7], vertices[3], vertices[0]]
#         ]

#         box = Poly3DCollection(edges, alpha=0.1, facecolors='white', linewidths=0.5, edgecolors='k')
#         ax.add_collection3d(box)

#     def plot_items(ax, item_positions, padding):
#         colors = {
#             'L_B_H': 'blue',
#             'L_H_B': 'lightblue',
#             'B_L_H': 'grey',
#             'B_H_L': 'lightgreen',
#             'H_L_B': 'red',
#             'H_B_L': 'salmon'
#         }

#         for x, y, z, dim, orientation in item_positions:
#             item_color = colors[orientation]
#             item_coords = np.array([
#                 [x, y, z],
#                 [x + dim[0], y, z],
#                 [x + dim[0], y + dim[1], z],
#                 [x, y + dim[1], z],
#                 [x, y, z + dim[2]],
#                 [x + dim[0], y, z + dim[2]],
#                 [x + dim[0], y + dim[1], z + dim[2]],
#                 [x, y + dim[1], z + dim[2]]
#             ])

#             edges = [
#                 [item_coords[0], item_coords[1], item_coords[2], item_coords[3]],
#                 [item_coords[4], item_coords[5], item_coords[6], item_coords[7]],
#                 [item_coords[0], item_coords[1], item_coords[5], item_coords[4]],
#                 [item_coords[2], item_coords[3], item_coords[7], item_coords[6]],
#                 [item_coords[1], item_coords[2], item_coords[6], item_coords[5]],
#                 [item_coords[4], item_coords[7], item_coords[3], item_coords[0]]
#             ]

#             item = Poly3DCollection(edges, alpha=0.6, facecolors=item_color, linewidths=0.5, edgecolors='k')
#             ax.add_collection3d(item)

#     create_box(ax, L_box, B_box, H_box)
#     plot_items(ax, item_positions, padding)

#     ax.set_xlim(0, L_box)
#     ax.set_ylim(0, B_box)
#     ax.set_zlim(0, H_box)

#     ax.set_xlabel('Length (mm)')
#     ax.set_ylabel('Breadth (mm)')
#     ax.set_zlabel('Height (mm)')

#     ax.set_box_aspect([np.ptp([0, L_box]), np.ptp([0, B_box]), np.ptp([0, H_box])])

#     buffer = BytesIO()
#     plt.savefig(buffer, format='png')
#     buffer.seek(0)
#     img_str = base64.b64encode(buffer.read()).decode('utf-8')
#     plt.close(fig)
    
#     return img_str, total_items, total_weight, total_volume, orientation_count


CONTAINER_TRUCK_DIMENSIONS = {
    "Tempo_407": {"L_box": 2896, "B_box": 1676, "H_box": 1676, "max_weight": 2500},
    "13_Feet": {"L_box": 3962, "B_box": 1676, "H_box": 2134, "max_weight": 3500},
    "14_Feet": {"L_box": 4267, "B_box": 1829, "H_box": 1829, "max_weight": 4000},
    "17_Feet": {"L_box": 5182, "B_box": 1829, "H_box": 2134, "max_weight": 6000},
    "20_ft_sxl": {"L_box": 6096, "B_box": 2438, "H_box": 2438, "max_weight": 7000},
    "24_ft_sxl": {"L_box": 7315, "B_box": 2438, "H_box": 2438, "max_weight": 7000},
    "32_ft_sxl": {"L_box": 9754, "B_box": 2438, "H_box": 2438, "max_weight": 7000},
    "32_ft_sxl_HQ": {"L_box": 9754, "B_box": 2743, "H_box": 2896, "max_weight": 7000},
    "32_ft_mxl": {"L_box": 9754, "B_box": 2438, "H_box": 2438, "max_weight": 15000},
    "32_ft_mxl_HQ": {"L_box": 9754, "B_box": 2743, "H_box": 2896, "max_weight": 14000},
}

def get_box_dimensions(box_key):
    return CONTAINER_TRUCK_DIMENSIONS.get(box_key)