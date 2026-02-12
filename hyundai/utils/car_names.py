CAR_NAME_TO_EN = {
    "CE": "CE",
    "DF": "DF",
    "GN7 일반": "GN7_Normal",
    "GN7 파노라마": "GN7_Panorama",
}


def to_english_car_name(name):
    return CAR_NAME_TO_EN.get(name, name)
