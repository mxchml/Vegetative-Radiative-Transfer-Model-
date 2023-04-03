# Imports
import shutil
import shutil as sh  # Library containing many file manipulation functions.
import math  # This module provides access to the mathematical functions defined by the C standard.
import os  # provides a portable way of using operating system dependent functionality.
import random  # pseudo-random number generators for various distributions.
import time
import datetime
from sklearn.linear_model import LinearRegression

# See how long the program takes to run
import numpy as np
from scipy import integrate

pi = math.pi  # import the value of PI from the python math library.

start = time.time()

# Directories
input_dir = r"./inputs/"  # Where the geo files, mat files, and lamp data is stored
radiance_dir = r"./libraries/radiance/"  # where the radiance engine is stored
results_dir = r"./results/"  # where all outputs of this script are stored
bestSimplex_dir = r"./ParaFits/"

# Vertex
# 0: lai
# 1: canopy_hor_lad
# 2: canopy_vertical_lad

vertex_names = ["lai",
                " | HLAD",
                " | VLAD"]

leaf_abs = [0.950, 0.937, 0.969]

# lamp = [74.06, 12.60, 13.00]
lamp = [96.108, 13.825, 12.372]
# lamp = [90.2442, 11.466, 9.88]

wall = [0.95, 0.95, 0.78]
floor = [0.36, 0.34, 0.29]

def clear_output_directory():  # This code clears the contents of the output directory
    if os.path.exists(results_dir):  # Checks to see if the folder called 'output_files' exists
        sh.rmtree(results_dir)  # Deletes the 'output_files' folder
        os.mkdir(results_dir)  # Makes a new, and empty, the 'output_files' folder
    else:
        os.mkdir(results_dir)  # Makes a new, and empty, the 'output_files' folder

def clear_ParaFits():  # This code clears the contents of the output directory
    if os.path.exists(bestSimplex_dir):  # Checks to see if the folder called 'output_files' exists
        sh.rmtree(bestSimplex_dir)  # Deletes the 'output_files' folder
        os.mkdir(bestSimplex_dir)  # Makes a new, and empty, the 'output_files' folder
    else:
        os.mkdir(bestSimplex_dir)  # Makes a new, and empty, the 'output_files' folder

def saveBestSimplex(run_count, directory):
    filepath = directory + "best_simplexes.txt"

    now = datetime.datetime.now()
    timestamp = now.strftime(("%Y-%m-%d__%H-%M-%S"))
    export_line_0 = str(run_count) + " " + timestamp + " "

    performance_file = open("performance_file.txt", 'r')
    performance_file_contents = performance_file.readlines()
    performance_file.close()
    pf_len = len(performance_file_contents)
    export_line_1 = performance_file_contents[pf_len - 1].strip("\n")

    vertex = [1.1, 1.1, 1.1]
    vertex[0] = float(export_line_1.split(", ")[0])
    vertex[1] = float(export_line_1.split(", ")[1])
    vertex[2] = float(export_line_1.split(", ")[2])
    prep_simulation(vertex, r"./inputs_all/", directory)
    predictions = run_simulation_hq(radiance_dir, r"./inputs_all/", directory)

    measurements = import_measures(r"./inputs_all/" + "measurements.txt")

    R2 = calculate_r2(predictions, measurements)
    export_line_2 = str(R2)

    export_line = export_line_0 + export_line_1 + export_line_2 + "\n"

    file = open(filepath, 'a')
    file.write(export_line)
    file.close()

def generate_initial_vertex_values():
    return [random.uniform(2, 13),  # lai
            random.uniform(0.2, 2),
            random.uniform(0.2, 0.8)]

def modify_initial_vertex_values(parameters):
    output = []
    for i in range(len(parameters)):
        if i == 0:
            min = parameters[i] - 1
            max = parameters[i] + 1
        else:
            min = parameters[i] - 0.1
            max = parameters[i] + 0.1
        if min < 0:
            min = 0
        if max > 2 and i != 0:
            max = 2

        output.append(random.uniform(min, max))

    return output

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
    # Simplex boundary
    for item in range(len(vertex)):  # bound the vertex values to between 0 and 1 for all but the last vertex
        if vertex[item] <= 0:
            vertex[item] = 0.01
        if item == 0 and vertex[item] > 15:
            vertex[item] = 15
        if item == 1 and vertex[item] > 1000:
            vertex[item] = 1000
        if item == 2 and vertex[item] > 1:
            vertex[item] = 1

    # round values
        if item == 0:
            vertex[item] = round(vertex[item], 1)
        if item != 0:
            vertex[item] = round(vertex[item], 2)

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
    can_hor_lad = vertex[1]
    can_vert_lad = vertex[2]
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

    def mod_geo_file(canopy_geo):
        file = open(input_dir + "geometries.rad", 'r')
        geo_data = file.readlines()
        file.close()

        can_width = canopy_geo[1]

        for line in range(len(geo_data)):
            if "!" in geo_data[line] and "ext" in geo_data[line]:
                break

        new_geo_data = geo_data[0:line]

        str1 = "!libraries/radiance/bin/genbox canopy_ext_"
        str2 = " 2.746 "
        str3 = " 0.0797 | libraries/radiance/bin/xform -t "

        for row in range(6):
            for col in range(6):
                cube_ID = "r" + str(row + 1) + "c" + str(col + 1)
                canopy_ID = " canopy" + str(1) + "_"
                first_can_width = can_width - 0.2
                cube_w = str(first_can_width / 6)[0:6]

                x_loc = "0.2"
                y_loc = str((can_mid_loc[0] - (first_can_width / 2)) + (first_can_width * col / 6))[0:6]
                z_loc = str(1.179 + (0.478 * row / 6))[0:6]
                loc = " " + x_loc + " " + y_loc + " " + z_loc + " "

                string = str1 + cube_ID + canopy_ID + cube_ID + str2 + cube_w + str3 + loc + " \n"
                new_geo_data.append(string)

        new_geo_data.append(" \n \n")

        for can in range(1, len(can_mid_loc)):
            for row in range(6):
                for col in range(6):
                    cube_ID = "r" + str(row + 1) + "c" + str(col + 1)
                    canopy_ID = " canopy" + str(can + 1) + "_"
                    cube_w = str(can_width / 6)[0:6]

                    x_loc = "0.2"
                    if can >= 12:
                        x_loc = "4.66"
                    y_loc = str((can_mid_loc[can] - (can_width / 2)) + (can_width * col / 6))[0:6]
                    z_loc = str(1.179 + (0.478 * row / 6))[0:6]
                    loc = " " + x_loc + " " + y_loc + " " + z_loc + " "

                    string = str1 + cube_ID + canopy_ID + cube_ID + str2 + cube_w + str3 + loc + " \n"
                    new_geo_data.append(string)

            new_geo_data.append(" \n \n \n")

        new_geo_data.append("# Canopy reflection \n")

        str1 = "!libraries/radiance/bin/genbox canopy_ref canopyR"
        str2 = " 2.7462 "
        str3 = " 0.4782 | libraries/radiance/bin/xform -t "

        can_ID = str(1)
        canopy_width = str(first_can_width + 0.0002)[0:6]
        y_origin = str(can_mid_loc[0] - (first_can_width / 2))[0:6]

        x_loc = "0.1999"
        y_loc = str(can_mid_loc[0] - (first_can_width / 2) - 0.0001)[0:6]
        z_loc = "1.1789"
        loc = " " + x_loc + " " + y_loc + " " + z_loc

        string = str1 + can_ID + str2 + canopy_width + str3 + loc + " \n"
        new_geo_data.append(string)

        for can in range(1, len(can_mid_loc) - 16):
            can_ID = str(can + 1)
            canopy_width = str(can_width + 0.0002)[0:6]
            y_origin = str(can_mid_loc[can] - (can_width / 2))[0:6]

            x_loc = "0.1999"
            if can >= 12:
                x_loc = "4.6599"
            y_loc = str(can_mid_loc[can] - (can_width / 2) - 0.0001)[0:6]
            z_loc = "1.1789"
            loc = " " + x_loc + " " + y_loc + " " + z_loc + " "

            string = str1 + can_ID + str2 + canopy_width + str3 + loc + " \n"
            new_geo_data.append(string)

        file = open(input_dir + "geometries.rad", 'w')
        file.writelines(new_geo_data)
        file.close()

    mod_geo_file(canopy_geo)

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
        global lamp

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
    # # Max messing around
    # op1 = " -I -h -dp 1 -ds 0 -dt 1 -dc 0 -dr 1 -st 1 -lr 0 -lw 0.05 -ab 1 -ad 1"
    # op2 = " -as 0 -ar 1 -aa 0.5 -n 6 -e error.log "
    oct_file = output_dir + "octree.oct"
    test_grid = input_dir + "test_points.pts"
    string = execu + op1 + op2 + oct_file + " < " + test_grid + " > " + output_dir + "results.res"
    os.system(string)
    sh.move("error.log", output_dir + "/error.log")
    results = import_preds(output_dir + "results.res")

    # # RPICT
    # # −vp 10 5 3 −vd 1 −.5 0 scene.oct > scene.hdr
    # execu = radiance_dir + "bin/rpict"
    # op = " -vp 4.6 1 1 -vd -1 0 0 "
    # result_loc = output_dir + "results.hdr"
    # string = execu + op + oct_file + " > " + result_loc
    # os.system(string)
    os.remove("rayinit.cal")
    os.remove(oct_file)

    global simulation_counter
    simulation_counter = simulation_counter + 1

    end = time.time()
    duration = end - start
    print("duration:" + str(duration))

    return results

def run_simulation_hq(radiance_dir, input_dir, output_dir):
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
    # # Max messing around
    # op1 = " -I -h -dp 1 -ds 0 -dt 1 -dc 0 -dr 1 -st 1 -lr 0 -lw 0.05 -ab 1 -ad 1"
    # op2 = " -as 0 -ar 1 -aa 0.5 -n 6 -e error.log "
    oct_file = output_dir + "octree.oct"
    test_grid = input_dir + "test_points.pts"
    string = execu + op1 + op2 + oct_file + " < " + test_grid + " > " + output_dir + "results.res"
    os.system(string)
    sh.move("error.log", output_dir + "/error.log")
    results = import_preds(output_dir + "results.res")

    # # RPICT
    # # −vp 10 5 3 −vd 1 −.5 0 scene.oct > scene.hdr
    # execu = radiance_dir + "bin/rpict"
    # op = " -vp 4.6 1 1 -vd -1 0 0 "
    # result_loc = output_dir + "results.hdr"
    # string = execu + op + oct_file + " > " + result_loc
    # os.system(string)
    os.remove("rayinit.cal")
    os.remove(oct_file)

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

def write_last_preds(preds, output_folder):
    file = open("last_predictions.txt", 'w')

    for pt in range(len(preds)):
        red = str(preds[pt][0]) + " "
        green = str(preds[pt][1]) + " "
        blue = str(preds[pt][2]) + " "

        this_line = red + green + blue + " \n"

        file.write(this_line)

    file.close()


def write_preds(preds, output_folder):
    file = open(output_folder + "predictions" + str(simulation_counter) + ".txt", 'w')
    print("predictions" + str(simulation_counter) + ".txt")

    for pt in range(len(preds)):
        red = str(preds[pt][0]) + " "
        green = str(preds[pt][1]) + " "
        blue = str(preds[pt][2]) + " "

        this_line = red + green + blue + " \n"

        file.write(this_line)

    file.close()

    write_last_preds(preds, output_folder)

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

def calculate_r2(preds, measures):
    model = LinearRegression()

    predictions = []
    measurements = []

    for m in range(len(measures)):
        pred_total = preds[m][0] + preds[m][1] + preds[m][2]
        predictions.append(pred_total)

        measure_total = measures[m][0] + measures[m][1] + measures[m][2]
        measurements.append(measure_total)

    x = np.array(measurements).reshape(-1, 1)
    y = np.array(predictions)

    model.fit(x, y)

    r_squared = model.score(x, y)

    return r_squared

# Generate Starting Simplex Functions

def gen_first_iteration(measurements):
    simplex = []
    iteration_dir = results_dir + "I1/"
    os.mkdir(iteration_dir)

    global vertex_names

    initial_vertex_values = generate_initial_vertex_values()

    for v in range(len(vertex_names) + 1):  # Number of parameters + 1
        print("First Simplex, Vertex #: " + str(v))
        modified_vertex_values = modify_initial_vertex_values(initial_vertex_values)
        prep_simulation(modified_vertex_values, input_dir, iteration_dir)
        predictions = run_simulation(radiance_dir, input_dir, iteration_dir)
        write_preds(predictions, iteration_dir)

        rmse = calculate_rmse(predictions, measurements)  # calculation of the rmse value at a single vertex

        modified_vertex_values.append(rmse)  # attaching the computed rmse value to the vertex array
        simplex.append(modified_vertex_values)  # attaching the vertex to the matrix

    return simplex


def find_worst_index(simplex):
    # Identify the worst vertex
    unmodified_rmse = []
    for v in range(len(simplex)):
        last_item_index = len(simplex[v]) - 1
        unmodified_rmse.append(simplex[v][last_item_index])

    modified_rmse = []
    for item in range(len(unmodified_rmse)):
        modified_rmse.append(unmodified_rmse[item])

    modified_rmse.sort()

    worst_value = modified_rmse[len(modified_rmse) - 1]
    worst_index = unmodified_rmse.index(worst_value)

    return worst_index


def find_second_worst_index(simplex):
    # Identify the second-worst vertex
    unmodified_rmse = []
    for v in range(len(simplex)):
        last_item_index = len(simplex[v]) - 1
        unmodified_rmse.append(simplex[v][last_item_index])

    modified_rmse = []
    for item in range(len(unmodified_rmse)):
        modified_rmse.append(unmodified_rmse[item])

    modified_rmse.sort()

    second_worst_value = modified_rmse[len(modified_rmse) - 2]
    second_worst_index = unmodified_rmse.index(second_worst_value)

    return second_worst_index


def find_best_index(simplex):
    # Identify the best vertex
    unmodified_rmse = []
    for v in range(len(simplex)):
        last_item_index = len(simplex[v]) - 1
        unmodified_rmse.append(simplex[v][last_item_index])

    modified_rmse = []
    for item in range(len(unmodified_rmse)):
        modified_rmse.append(unmodified_rmse[item])

    modified_rmse.sort()

    best_value = modified_rmse[0]
    best_index = unmodified_rmse.index(best_value)

    return best_index


def remove_worst_vertex(simplex):  # removes the vertex with the highest rmse value from the matrix
    new_iteration_simplex = []

    worst_index = find_worst_index(simplex)

    # Sum number of vertices
    for v in range(len(simplex)):
        this_vertex = []
        if v == worst_index:
            continue
        else:
            for item in range(len(simplex[0])):
                this_vertex.append(simplex[v][item])

        new_iteration_simplex.append(this_vertex)

    return new_iteration_simplex


def calculate_centroid(simplex):  # Calculates the centroid of the simplex
    centroid = []

    for item in range(len(simplex[0]) - 1):
        vertices_item_sum = 0

        for v in range(len(simplex)):
            vertices_item_sum = vertices_item_sum + simplex[v][item]

        centroid_item = vertices_item_sum / (len(simplex[0]) - 1)

        centroid.append(centroid_item)

    return centroid


def compute_reflected_point(simplex, measurements, iteration_dir):
    print("returning reflected point")
    reflection_point = []

    index = find_worst_index(simplex)
    worst_vertex = simplex[index]

    simplex_with_removed_vertex = remove_worst_vertex(simplex)

    centroid = calculate_centroid(simplex_with_removed_vertex)

    # compute the reflection point
    for item in range(len(centroid)):
        this = centroid[item] + reflection_parameter * (centroid[item] - worst_vertex[item])
        reflection_point.append(this)

    prep_simulation(reflection_point, input_dir, iteration_dir)
    predictions = run_simulation(radiance_dir, input_dir, iteration_dir)
    write_preds(predictions, iteration_dir)

    rmse = calculate_rmse(predictions, measurements)
    reflection_point.append(rmse)

    return reflection_point


def compute_expanded_point(simplex, reflected_point, measurements, iteration_dir):
    print("returning expansion point")
    expansion_point = []

    simplex_without_worst_vertex = remove_worst_vertex(simplex)

    centroid = calculate_centroid(simplex_without_worst_vertex)

    # compute the reflection point
    for item in range(len(centroid)):
        this = centroid[item] + expansion_parameter * (reflected_point[item] - centroid[item])
        expansion_point.append(this)

    prep_simulation(expansion_point, input_dir, iteration_dir)
    predictions = run_simulation(radiance_dir, input_dir, iteration_dir)
    write_preds(predictions, iteration_dir)

    rmse = calculate_rmse(predictions, measurements)
    expansion_point.append(rmse)

    return expansion_point


def compute_contracted_point_outside(simplex, reflected_point, measurements, iteration_dir):
    print("returning outside contracted point")
    contraction_point = []
    simplex_without_worst_vertex = remove_worst_vertex(simplex)
    centroid = calculate_centroid(simplex_without_worst_vertex)

    # compute the reflection point
    for item in range(len(centroid)):
        this = centroid[item] + contraction_parameter * (reflected_point[item] - centroid[item])
        contraction_point.append(this)

    prep_simulation(contraction_point, input_dir, iteration_dir)
    predictions = run_simulation(radiance_dir, input_dir, iteration_dir)
    write_preds(predictions, iteration_dir)

    rmse = calculate_rmse(predictions, measurements)
    contraction_point.append(rmse)

    return contraction_point


def compute_contracted_point_inside(simplex, measurements, iteration_dir):
    print("returning inside contracted point")
    contraction_point = []

    index = find_worst_index(simplex)
    worst_vertex = simplex[index]

    simplex_without_worst_vertex = remove_worst_vertex(simplex)

    centroid = calculate_centroid(simplex_without_worst_vertex)

    # compute the reflection point
    for item in range(len(centroid)):
        this = centroid[item] + contraction_parameter * (worst_vertex[item] - centroid[item])
        contraction_point.append(this)

    prep_simulation(contraction_point, input_dir, iteration_dir)
    predictions = run_simulation(radiance_dir, input_dir, iteration_dir)
    write_preds(predictions, iteration_dir)

    rmse = calculate_rmse(predictions, measurements)
    contraction_point.append(rmse)

    return contraction_point


def compute_shrink_matrix(simplex, measurements, iteration_dir):
    print("returning shrink matrix")
    shrink_matrix = []

    best_index = find_best_index(simplex)
    best_vertex = simplex[best_index]

    for v in range(len(simplex)):
        if v == best_index:
            shrink_matrix.append(best_vertex)
        else:
            this_vertex = []
            for item in range(len(best_vertex) - 1):
                this_vertex.append(
                    best_vertex[item] + shrinkage_parameter * (simplex[v][item] - best_vertex[item]))

            prep_simulation(this_vertex, input_dir, iteration_dir)
            predictions = run_simulation(radiance_dir, input_dir, iteration_dir)
            write_preds(predictions, iteration_dir)

            rmse = calculate_rmse(predictions, measurements)
            this_vertex.append(rmse)

            shrink_matrix.append(this_vertex)

    return shrink_matrix


# Nelder Mead
def nelder_mead_single_iteration(iter_count):
    iteration_dir = results_dir + "I" + str(iter_count) + "/"
    file_path = iteration_dir + "iteration_simplex.txt"
    measurements = import_measures(input_dir + "measurements.txt")

    simplex = load_simplex_from_file(file_path)

    # Prepare
    best_index = find_best_index(simplex)
    second_worst_index = find_second_worst_index(simplex)
    worst_index = find_best_index(simplex)

    best_vertex = simplex[best_index]
    second_worst_vertex = simplex[second_worst_index]
    worst_vertex = simplex[worst_index]

    def return_vertex_rmse(vertex):
        return vertex[len(vertex) - 1]

    best_vertex_rmse = return_vertex_rmse(best_vertex)
    second_worst_vertex_rmse = return_vertex_rmse(second_worst_vertex)
    worst_vertex_rmse = return_vertex_rmse(worst_vertex)
    reflected_point = compute_reflected_point(simplex, measurements, iteration_dir)
    reflected_point_rmse = reflected_point[len(reflected_point) - 1]

    if best_vertex_rmse <= reflected_point_rmse < second_worst_vertex_rmse:
        new_iteration_simplex = remove_worst_vertex(simplex)
        new_iteration_simplex.append(reflected_point)
        return new_iteration_simplex

    elif reflected_point_rmse < best_vertex_rmse:
        expanded_point = compute_expanded_point(simplex, reflected_point, measurements, iteration_dir)
        expanded_point_rmse = expanded_point[len(expanded_point) - 1]

        if expanded_point_rmse < reflected_point_rmse:
            new_iteration_simplex = remove_worst_vertex(simplex)
            new_iteration_simplex.append(expanded_point)
            return new_iteration_simplex

        else:
            new_iteration_simplex = remove_worst_vertex(simplex)
            new_iteration_simplex.append(reflected_point)
            return new_iteration_simplex

    elif reflected_point_rmse >= second_worst_vertex_rmse:
        if second_worst_vertex_rmse <= reflected_point_rmse < worst_vertex_rmse:
            contract_point = compute_contracted_point_outside(simplex, reflected_point, measurements, iteration_dir)
            contract_point_rmse = contract_point[len(contract_point) - 1]
            if contract_point_rmse <= reflected_point_rmse:
                new_iteration_simplex = remove_worst_vertex(simplex)
                new_iteration_simplex.append(contract_point)
                return new_iteration_simplex
            else:
                return compute_shrink_matrix(simplex, measurements, iteration_dir)

        elif reflected_point_rmse >= worst_vertex_rmse:
            contract_point = compute_contracted_point_inside(simplex, measurements, iteration_dir)
            contract_point_rmse = contract_point[len(contract_point) - 1]
            if contract_point_rmse < reflected_point_rmse:
                new_iteration_simplex = remove_worst_vertex(simplex)
                new_iteration_simplex.append(contract_point)
                return new_iteration_simplex
            else:
                return compute_shrink_matrix(simplex, measurements, iteration_dir)

        else:
            return compute_shrink_matrix(simplex, measurements, iteration_dir)


def compute_average_vertex_of_simplex(simplex):
    averages = []
    for item in range(len(simplex[0])):
        sum = 0
        for v in range(len(simplex)):
            sum = sum + simplex[v][item]
        this_average = sum / len(simplex[v])
        averages.append(this_average)

    return averages


def create_performance_file(output_folder):
    file_path = "performance_file.txt"
    file = open(file_path, 'w')
    file.close()
    return file_path


def append_performance_file(perform_path, iteration_dir):
    # import into simplex array for easy handling
    simplex = load_simplex_from_file(iteration_dir + "iteration_simplex.txt")

    RMSEs = []
    for v in range(len(simplex)):
        RMSEs.append(simplex[v][len(simplex[v]) - 1])

    min_RMSE = min(RMSEs)
    min_index = RMSEs.index(min_RMSE)

    string = ""
    for item in range(len(simplex[min_index])):
        if item != len(simplex[min_index]) - 1:
            string = string + str(simplex[min_index][item])[0:8] + ", "
        else:
            string = string + str(simplex[min_index][item]) + ",  "


    file = open(perform_path, 'a')
    file.write(string + "\n")
    file.close()

def domain_termination_test(iteration_number):
    previous_simplex = load_simplex_from_file(results_dir + "I" + str(iteration_number) + "/iteration_simplex.txt")
    current_simplex = load_simplex_from_file(results_dir + "I" + str(iteration_number + 1) + "/iteration_simplex.txt")

    truth_array = []

    for v in range(len(previous_simplex[0])):
        for item in range(len(previous_simplex[v]) - 1):
            difference = abs(previous_simplex[v][item] - current_simplex[v][item])
            if difference < (10 ** -4) * (1 + abs(current_simplex[v][item])):
                truth_array.append(True)
            else:
                truth_array.append(False)

    if all(truth_array):
        global domain_termination
        domain_termination = 1


def function_value_termination_test(iteration_number):
    previous_simplex = load_simplex_from_file(results_dir + "I" + str(iteration_number) + "/iteration_simplex.txt")
    current_simplex = load_simplex_from_file(results_dir + "I" + str(iteration_number + 1) + "/iteration_simplex.txt")

    truth_array = []

    for v in range(len(previous_simplex[0])):
        rmse_loc = len(previous_simplex[v]) - 1
        difference = abs(previous_simplex[v][rmse_loc] - current_simplex[v][rmse_loc])
        if difference < (10 ** -3) * (1 + abs(current_simplex[v][rmse_loc])):
            truth_array.append(True)
        else:
            truth_array.append(False)

    if all(truth_array):
        global function_termination
        function_termination = 1


def write_simplex(simplex, directory):
    # Write the newly generated simplex to the 'iteration_simplex.txt' file

    file = open(directory + "iteration_simplex.txt", 'w')

    for v in range(len(simplex)):
        for item in range(len(simplex[v])):
            if item != len(simplex[v]) - 1:
                string = str(simplex[v][item])[0:8] + " "
            else:
                string = str(simplex[v][item]) + " "
            file.write(string)

        file.write(" \n")

    file.close()
    shutil.copy(directory + "iteration_simplex.txt", "iteration_simplex.txt")


def nelder_mead_iterator(number_of_iterations):
    global domain_termination
    global function_termination

    perform_path = create_performance_file(results_dir)  # function that creates performance_file.txt

    for iteration in range(1, number_of_iterations + 1):

        print("\n " + " Iteration #:" + str(iteration))

        # the loop that causes the Nelder Mead algorithm to begin iteration
        iteration_start = time.time()
        iteration_simplex = nelder_mead_single_iteration(iteration)  # function that executes a single iteration
        iteration_duration = time.time() - iteration_start
        print("Iteration duration: " + str(iteration_duration)[0:7])

        next_dir = results_dir + "I" + str(iteration + 1) + "/"
        os.mkdir(next_dir)
        write_simplex(iteration_simplex, next_dir)

        averages = compute_average_vertex_of_simplex(iteration_simplex)
        # function that generates the data for the performance_file.txt

        append_performance_file(perform_path, next_dir)
        # function that writes the data to performance_file.txt

        if iteration > 1:
            domain_termination_test(iteration)
            # Function that checks domain termination condition
            function_value_termination_test(iteration)
            # Function that checks function RMSE termination condition

        if domain_termination == 1:
            print("Nelder Mead terminated due to domain termination")
            domain_termination = 0
            return 1

        elif function_termination == 1:
            print("Nelder Mead terminated due to function value termination")
            function_termination = 0
            return 1

        elif iteration == number_of_iterations:
            print("Nelder Mead  Terminated due to iteration counter reaching limit")
            domain_termination = 0
            function_termination = 0
            return 1

vertex_holder = []

clear_ParaFits()

for i in range(0, 50):
    # Activate Program
    clear_output_directory()  # clear all folders from previous runnings of the algorithm

    measurements = import_measures(input_dir + "measurements.txt")
    initial_simplex = gen_first_iteration(measurements)  # generate the starting simplex data.
    # initial_simplex = load_simplex_from_file(results_dir + "I1/" + "iteration_simplex.txt")
    compute_average_vertex_of_simplex(initial_simplex)
    write_simplex(initial_simplex, results_dir + "I1/")  # write the starting simplex data to the output_files folder

    nelder_mead_iterator(500)
    saveBestSimplex(i, bestSimplex_dir)

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


