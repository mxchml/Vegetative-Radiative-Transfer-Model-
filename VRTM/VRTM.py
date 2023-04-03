# Imports
import shutil
import shutil as sh  # Library containing many file manipulation functions.
import math  # This module provides access to the mathematical functions defined by the C standard.
import os  # provides a portable way of using operating system dependent functionality.
import random  # pseudo-random number generators for various distributions.
import time
import datetime

# See how long the program takes to run
import numpy as np
from scipy import integrate

pi = math.pi  # import the value of PI from the python math library.

start = time.time()

# Directories
input_dir = r"./inputs/"  # Where the geo files, mat files, and lamp data is stored
radiance_dir = r"./libraries/radiance/"  # where the radiance engine is stored
results_dir = r"./results/"  # where all outputs of this script are stored

LAI = 15
leaf_abs = [0.950, 0.937, 0.969]
hor = 1.3
vert = 2.73
vertex = [LAI, leaf_abs[0], leaf_abs[1], leaf_abs[2], hor, vert]

lamp = [74.06, 12.60, 13.00]
# lamp = [96.108, 13.825, 12.372]
# lamp = [90.2442, 11.466, 9.88]

# wall = [0.95, 0.95, 0.78]
# floor = [0.36, 0.34, 0.29]

wall = [1, 1, 1]
floor = [0.36, 0.34, 0.29]

vertex_names = ["lai",
                " | abs_red",
                " | abs_green",
                " | abs_blue",
                " | HLAD",
                " | VLAD"]

def clear_output_directory():  # This code clears the contents of the output directory
    if os.path.exists(results_dir):  # Checks to see if the folder called 'output_files' exists
        sh.rmtree(results_dir)  # Deletes the 'output_files' folder
        os.mkdir(results_dir)  # Makes a new, and empty, the 'output_files' folder
    else:
        os.mkdir(results_dir)  # Makes a new, and empty, the 'output_files' folder

def savePerformanceFile():
    now = datetime.datetime.now()
    timestamp = now.strftime(("%Y-%m-%d__%H-%M-%S"))

    filepath = "/Users/mxchml/Desktop/performance_file_" + timestamp + ".txt"
    print(filepath)
    shutil.copy2("performance_file.txt", filepath)

    filepath = "/Users/mxchml/Desktop/predictions_" + timestamp + ".txt"
    shutil.copy2("current_predictions.txt", filepath)

can_mid_loc = [0.8, 1.853, 2.854, 3.96, 5.009, 6.035, 7.122, 8.232,
               9.148, 10.235, 11.316, 12.277, 0.719, 1.793, 2.85, 3.884,
               4.925, 5.975, 7.086, 8.188, 9.24, 10.355, 11.464, 12.365]

# Global variables
n = len(vertex_names)  # Number of parameters in a Simplex vertex
iteration_counter = 1  # variable that tracks the number of iterations
reflection_parameter = 1  # The Nelder Mead reflection parameter
contraction_parameter = 0.75 - (1 / (2 * n))  # The Nelder Mead contraction parameter
expansion_parameter = 1 + (2 / n)  # The Nelder Mead expansion_parameter parameter
shrinkage_parameter = 1 - (1 / n)  # The Nelder Mead expansion_parameter parameter

domain_termination = 0
function_termination = 0
simulation_counter = 0

# Memory Management Functions
def clear_output_directory():  # This code clears the contents of the output directory
    if os.path.exists(results_dir):  # Checks to see if the folder called 'output_files' exists
        sh.rmtree(results_dir)  # Deletes the 'output_files' folder
        os.mkdir(results_dir)  # Makes a new, and empty, the 'output_files' folder
    else:
        os.mkdir(results_dir)  # Makes a new, and empty, the 'output_files' folder

def load_simplex_from_file(file_path):  # Loads the data stored in the 'iteration_simplex.txt' file
    file = open(file_path, 'r')
    file_contents = file.readlines()
    file.close()

    simplex = []

    for line in range(len(file_contents)):
        this_line = file_contents[line].split()
        this_line_float = []

        for item in range(len(this_line)):
            this_line_float.append(float(this_line[item]))

        simplex.append(this_line_float)

    return simplex


def prep_simulation(vertex, input_dir, output_dir):

    vertex_for_print = ""
    for item in range(len(vertex)):
        this = vertex[item]
        vertex_for_print = vertex_for_print + vertex_names[item] + ":" + str(this)[0:6]
    print(vertex_for_print)

    # LAI
    lai = vertex[0]

    # Canopy geometry wrapped into an array
    can_height = 0.478
    can_width = 0.684
    can_hor_lad = vertex[4]
    can_vert_lad = vertex[5]
    canopy_geo = [can_height, can_width, can_hor_lad, can_vert_lad]

    # Calculation of the angle extinction based on the angle_ratio parameter
    angle_ratio = 1
    zenith_angle = math.atan(0.684 / 0.478)

    horizontal_angle = 90 - zenith_angle
    numerator_vertical = math.sqrt(angle_ratio ** 2 + math.tan(zenith_angle) ** 2)
    numerator_horizontal = math.sqrt(angle_ratio ** 2 + math.tan(horizontal_angle) ** 2)

    angle_ext_denom = angle_ratio + 1.774 * (angle_ratio + 1.182) ** -0.733
    angle_ext_vertical = numerator_vertical / angle_ext_denom
    angle_ext_horizontal = numerator_horizontal / angle_ext_denom

    angle_ext = [angle_ext_vertical, angle_ext_horizontal]

    # Canopy leaf morphological and optical properties
    global leaf_abs
    leaf_properties = [lai, angle_ext, leaf_abs]

    def compute_lad_hor(canopy_geo, leaf_properties):
        result = []

        can_width = canopy_geo[1]
        lai = leaf_properties[0]
        can_lad_delta = canopy_geo[2]

        for cube in range(6):  # Stepping through each of the six cube columns in a single hedgerow
            f = lambda x: lai * (pi / 2 * can_width) * (np.sin(x * pi / can_width) ** can_lad_delta)
            lower_bound = (cube / 6) * can_width
            upper_bound = ((cube + 1) / 6) * can_width

            lad_hor = integrate.quad(f, lower_bound, upper_bound)[0]

            result.append(lad_hor)

        return result

    def compute_lad_vert(canopy_geo, leaf_properties):
        result = []

        can_height = canopy_geo[0]
        max_lad_frac = canopy_geo[3]
        lai = leaf_properties[0]

        lad_max_h = max_lad_frac * can_height

        for cube in range(6):
            n = (can_height / lad_max_h) ** 2
            denom1 = math.sqrt(pi / n)
            denom2 = math.erf(math.sqrt(n))
            denom3 = math.exp(-1 * n)
            denom = 0.5 * (0.5 * denom1 * denom2 - denom3)
            f = lambda z: ((lai * n * ((z / can_height) ** 2)) / denom) * math.exp(-1 * n * ((z / can_height) ** 2))

            lower_bound = (cube / 6) * can_height
            upper_bound = ((cube + 1) / 6) * can_height
            lad_vert = integrate.quad(f, lower_bound, upper_bound)[0]

            result.append(lad_vert)

        return result

    def compute_cube_lads(canopy_geo, leaf_properties):
        result = []

        for col in range(6):
            lad_hor = compute_lad_hor(canopy_geo, leaf_properties)
            lad_vert = compute_lad_vert(canopy_geo, leaf_properties)

            row_lads = []
            for row in range(6):
                cube_lad = [lad_vert[row], lad_hor[col]]
                row_lads.append(cube_lad)

            result.append(row_lads)

        return result

    def compute_cube_exts(canopy_geo, leaf_properties):  # use the computed LADs to compute each cube's optical extinction.
        result = []
        height = canopy_geo[0] / 6
        width = canopy_geo[1] / 6
        cdiag = math.sqrt(height ** 2 + width ** 2)
        angle_ext = leaf_properties[1]
        red_abs = leaf_properties[2][0]
        green_abs = leaf_properties[2][1]
        blue_abs = leaf_properties[2][2]

        cube_lads = compute_cube_lads(canopy_geo, leaf_properties)

        for col in range(6):
            row_exts = []

            for row in range(6):
                red_ext = (math.sqrt(red_abs) * angle_ext[0] * cube_lads[col][row][0] + \
                           math.sqrt(red_abs) * angle_ext[1] * cube_lads[col][row][1]) / cdiag
                green_ext = (math.sqrt(green_abs) * angle_ext[0] * cube_lads[col][row][0] + \
                             math.sqrt(green_abs) * angle_ext[1] * cube_lads[col][row][1]) / cdiag
                blue_ext = (math.sqrt(blue_abs) * angle_ext[0] * cube_lads[col][row][0] + \
                            math.sqrt(blue_abs) * angle_ext[1] * cube_lads[col][row][1]) / cdiag
                row_exts.append([red_ext, green_ext, blue_ext])


            result.append(row_exts)

        return result

    def write_cal_file(output_dir):
        brtd_path = output_dir + "BRTD.cal"

        file = open(brtd_path, 'w')  # generate new BRTDcanopy#.cal files

        file.write("trans_red = if (Nz, 1, 1);" + " \n")
        file.write("trans_green = if (Nz, 1, 1);" + " \n")
        file.write("trans_blue = if (Nz, 1, 1);" + " \n")

        file.close()

    def write_vertex_material_file(vertex, cube_ext, input_dir, output_dir):
        global wall
        global floor

        object_r = str(wall[0])
        object_g = str(wall[1])
        object_b = str(wall[2])
        wall_r = str(wall[0])
        wall_g = str(wall[1])
        wall_b = str(wall[2])
        floor_r = str(floor[0])
        floor_g = str(floor[1])
        floor_b = str(floor[2])

        ref_dp = []
        for color in range(3):
            ref_this_dp = (1 - math.sqrt(leaf_abs[color])) / (1 + math.sqrt(leaf_abs[color]))
            ref_this_dp_str = str(ref_this_dp)
            ref_dp.append(ref_this_dp_str)

        file = open(input_dir + "materials.rad", 'w')

        file.write("void BRTDfunc canopy_ref" + " \n")
        file.write("10 0 0 0 trans_red trans_green trans_blue 0 0 0 " + output_dir + "BRTD.cal \n")
        file.write("0 \n")
        file.write("9 " + str(ref_dp[0]) + " " + str(ref_dp[1]) + " " + str(ref_dp[2]) + " 0 0 0 0 0 0 \n")
        file.write("  \n")

        for col in range(6):
            for row in range(6):
                file.write("void mist " + "canopy_ext_r" + str(row + 1) + "c" + str(col + 1) + " \n")
                file.write("0 \n")
                file.write("0 \n")
                file.write("3 " + str(cube_ext[col][row][0]) + " " + str(cube_ext[col][row][1]) + " " + str(
                    cube_ext[col][row][2]))
                file.write("  \n")
                file.write("  \n")

        file.write("void plastic object \n")
        file.write("0 \n")
        file.write("0 \n")
        file.write("5 " + object_r + " " + object_g + " " + object_b + " 0 0 \n")
        file.write("  \n")

        file.write("void plastic wall \n")
        file.write("0 \n")
        file.write("0 \n")
        file.write("5 " + wall_r + " " + wall_g + " " + wall_b + " 0 0 \n")
        file.write("  \n")

        file.write("void plastic floor \n")
        file.write("0 \n")
        file.write("0 \n")
        file.write("5 " + floor_r + " " + floor_g + " " + floor_b + " 0 0 \n")
        file.write("  \n")

        file.write("void brightdata vividgro_dist \n")
        file.write("5 flatcorr inputs/vividgro_dist_mod.dat libraries/radiance/lib/source.cal src_phi src_theta \n")
        file.write("0 \n")
        file.write("1 1.54 \n")

        file.write("\n")

        file.write("vividgro_dist light vividgro \n")
        file.write("0 \n")
        file.write("0 \n")
        file.write("3 " + str(lamp[0]) + " " + str(lamp[1]) + " " + str(lamp[2]) + "  \n")
        file.write("\n")

        file.close()

    cube_exts = compute_cube_exts(canopy_geo, leaf_properties)
    write_cal_file(output_dir)
    write_vertex_material_file(vertex, cube_exts, input_dir, output_dir)

def run_simulation(radiance_dir, input_dir, output_dir):
    start = time.time()
    sh.copy("libraries/radiance/lib/rayinit.cal", "rayinit.cal")

    execu = radiance_dir + "bin/oconv "
    op1 = " -r 2048 -f "
    mat_path = input_dir + "materials.rad "
    geo_path = input_dir + "geometries.rad "
    string = execu + op1 + mat_path + geo_path + " > " + output_dir + "octree.oct"
    os.system(string)
    execu = radiance_dir + "bin/rtrace"
    # Correct Options
    op1 = " -I -h -dp 512 -ds 0.15 -dt 0.05 -dc 0.5 -dr 3 -st 0.15 -lr 8 -lw 0.002 -ab 5 -ad 512"
    op2 = " -as 256 -ar 300 -aa 0.1 -n 6 -e error.log "
    # # Ward accurate
    # op1 = " -I -h -dp 512 -ds 0.15 -dt 0.05 -dc 0.5 -dr 3 -st 0.15 -lr 8 -lw 0.002 -ab 4 -ad 512"
    # op2 = " -as 256 -ar 128 -aa 0.15 -n 6 -e error.log "
    # # Fast Options
    # op1 = " -I -h -dp 32 -ds 0 -dt 1 -dc 0 -dr 1 -st 1 -lr 0 -lw 0.05 -ab 4 -ad 32"
    # op2 = " -as 0 -ar 8 -aa 0.5 -n 6 -e error.log "
    # SUPER Fast Options
    # op1 = " -I -h -dp 32 -ds 0 -dt 1 -dc 0 -dr 0 -st 1 -lr 0 -lw 0.05 -ab 4 -ad 0"
    # op2 = " -as 0 -ar 8 -aa 0.5 -n 6 -e error.log "
    # # Max messing around
    # op1 = " -I -h -dp 128 -ds 0.15 -dt 0.15 -dc 0.5 -dr 3 -st 0.15 -lr 8 -lw 0.002 -ab 4 -ad 32"
    # op2 = " -as 16 -ar 32 -aa 0.15 -n 6 -e error.log "
    oct_file = output_dir + "octree.oct"
    test_grid = input_dir + "test_points.pts"
    string = execu + op1 + op2 + oct_file + " < " + test_grid + " > " + output_dir + "results.res"
    os.system(string)
    sh.move("error.log", output_dir + "/error.log")
    results = import_preds(output_dir + "results.res")

    # # RPICT
    # −vp 10 5 3 −vd 1 −.5 0 scene.oct > scene.hdr
    # execu = radiance_dir + "bin/rpict"
    # op = " -vp 4.6 1 1 -vd -1 0 0 "
    # op1 = " -dp 512 -ds 0.15 -dt 0.05 -dc 0.5 -dr 3 -st 0.15 -lr 8 -lw 0.002 -ab 5 -ad 512"
    # op2 = " -as 256 -ar 300 -aa 0.1 -e error.log "
    # result_loc = output_dir + "results.hdr"
    # string = execu + op + op1 + op2 + oct_file + " > " + result_loc
    # os.system(string)
    # os.remove("rayinit.cal")
    # os.remove(oct_file)
    # exit()

    global simulation_counter
    simulation_counter = simulation_counter + 1

    end = time.time()
    duration = end - start
    print("duration:" + str(duration))

    return results

# Data Management Functions
def import_points(path):
    file = open(path, 'r')
    file_content = file.readlines()
    file.close()

    loc = []

    for line in range(len(file_content)):
        this_line = file_content[line].split()
        loc.append([float(this_line[0]), float(this_line[1]), float(this_line[2]),
                    float(this_line[3]), float(this_line[4]), float(this_line[5])])

    return loc

def import_measures(path):  # import the measurements.txt from the 'measurements.txt' file
    file = open(path, 'r')
    file_content = file.readlines()
    file.close()

    intensities = []  # an array that takes the RGB irradiance values of the node

    for line in range(len(file_content)):
        this_line = file_content[line].split()
        intensities.append([float(this_line[3]), float(this_line[4]), float(this_line[5])])

    return intensities


def write_preds(preds, output_folder):
    filepath = output_folder + "predictions" + str(simulation_counter) + ".txt"
    file = open(filepath, 'w')
    print("predictions" + str(simulation_counter) + ".txt")

    for pt in range(len(preds)):
        red = str(preds[pt][0]) + " "
        green = str(preds[pt][1]) + " "
        blue = str(preds[pt][2]) + " "

        this_line = red + green + blue + " \n"

        file.write(this_line)

    file.close()

    shutil.copy2(filepath, "current_predictions.txt")

def import_preds(path):  # Import the simulation predictions into python
    file = open(path, 'r')
    file_contents = file.readlines()
    file.close()

    preds = []

    for line in range(len(file_contents)):
        this_line = file_contents[line].split()
        preds.append([float(this_line[0]), float(this_line[1]), float(this_line[2])])

    return preds

def calculate_rmse(preds, measures):
    difference_summation = 0

    for m in range(len(measures)):
        diff_red = (preds[m][0] - measures[m][0]) ** 2
        diff_green = (preds[m][1] * (lamp[0]/lamp[1]) - measures[m][1] * (lamp[0]/lamp[1])) ** 2
        diff_blue = (preds[m][2] * (lamp[0]/lamp[2]) - measures[m][2] * (lamp[0]/lamp[2])) ** 2
        difference_summation = difference_summation + diff_red + diff_green + diff_blue

    result = math.sqrt(difference_summation)
    print("   RMSE:" + str(result)[0:7])
    return result

clear_output_directory()  # clear all folders from previous runnings of the algorithm

measurements = import_measures(input_dir + "measurements.txt")
prep_simulation(vertex, input_dir, results_dir)
results = run_simulation(radiance_dir, input_dir, results_dir)
predictions = import_measures("inputs/measurements.txt")
RMSE = calculate_rmse(predictions, results)

savePerformanceFile()

stop = time.time()

print("start")
print(start)
print("stop")
print(stop)
print("time taken")
program_time = (stop - start) / 60
print(program_time)

file = open("performance_file.txt", 'a')
file.write("\n")
file.write("Time taken = " + str(program_time) + " minutes")
file.close()


