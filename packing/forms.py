from django import forms

CONTAINER_TRUCK_CHOICES = [
    ("Tempo_407", "Tempo 407"),
    ("13_Feet", "13 Feet"),
    ("14_Feet", "14 Feet"),
    ("17_Feet", "17 Feet"),
    ("20_ft_sxl", "20 ft SXL"),
    ("24_ft_sxl", "24 ft SXL"),
    ("32_ft_sxl", "32 ft SXL"),
    ("32_ft_sxl_HQ", "32 ft SXL HQ"),
    ("32_ft_mxl", "32 ft MXL"),
    ("32_ft_mxl_HQ", "32 ft MXL HQ"),
]

class SingleItemForm(forms.Form):
    truck_type = forms.ChoiceField(choices=CONTAINER_TRUCK_CHOICES, label="Truck Type")
    length = forms.FloatField(required=False)
    breadth = forms.FloatField(required=False)
    height = forms.FloatField(required=False)
    weight = forms.FloatField(required=False)
    orientations = forms.MultipleChoiceField(
        choices=[
            ("L_B_H", "L_B_H"),
            ("L_H_B", "L_H_B"),
            ("H_B_L", "H_B_L"),
            ("H_L_B", "H_L_B"),
            ("B_L_H", "B_L_H"),
            ("B_H_L", "B_H_L"),
        ],
        widget=forms.CheckboxSelectMultiple,
        label="Allowed Orientations",
        initial=["L_B_H"],
    )

    def clean(self):
        cleaned_data = super().clean()
        return cleaned_data


class MultipleItemsForm(forms.Form):
    pass