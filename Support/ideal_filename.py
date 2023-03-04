# -*- coding: utf-8 -*-
"""
|/--------------------TITLE--------------------\|
PHYS20161 -- Assignment N -- [ASSIGNMENT NAME]
|/---------------------------------------------\|

[CODE DESCRIPTION/USAGE/OUTLINE]


Created: Fri Mar  3 16:38:57 2023
Last Updated:

@author: Charlie Fynn Perkins, UID: 10839865 0
"""
import pathlib as Path


def filename(ideal_filename, extension):
    filename = ideal_filename
    address = "./RK4_Output/"  # Folder Address
    number = int(0)
    path = Path(address + filename + extension)  # Initial Path

    while path.is_file():
        number = number + int(1)
        number_append = str(number)
        filename = ideal_filename + '_' + number_append
        path = Path(address + filename + extension)
    return address + filename + extension
