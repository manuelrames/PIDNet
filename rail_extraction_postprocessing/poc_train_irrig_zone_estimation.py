import cv2
import numpy as np
import glob, math, os

import sys
sys.path.insert(1, '..')

from tools.custom import parse_args, inference, generate_colored_preds

# Fx = F/px where px = sensor_width [mm] / img_width [pix] (same applicable to Fy)
Fx = 336  # [pixels]
KNOWN_DISTANCE = 1668  # interail distance in Spain
WIDTH_IRRIG_ZONE = 2800  # irrigation zone width in [mm]


def find_unit_vector(point1, point2):
    dist = [int(point1[0][0]) - int(point2[0][0]), int(point1[1][0]) - int(point2[1][0])]
    norm = math.sqrt(dist[0] ** 2 + dist[1] ** 2)
    unitary_vector = [dist[0] / norm1, dist[1] / norm]

    return unitary_vector

def calc_line_slope(image, fit_line):
    # compute t0 for y=0 and t1 for y=img.shape[0]: (y-y0)/vy
    t0 = (0 - fit_line[3]) / fit_line[1]
    t1 = (image.shape[0] - fit_line[3]) / fit_line[1]

    # plug into the line formula to find the two endpoints, p0 and p1
    # to plot, we need pixel locations so convert to int
    p0 = (fit_line[2:4] + (t0 * fit_line[0:2])).astype(np.uint32)
    p1 = (fit_line[2:4] + (t1 * fit_line[0:2])).astype(np.uint32)

    # line slope (m = (y1-y0)/(x1-x0))
    # TODO denominator can be zero -> slope is infinite, so we aproximate it to 10000
    try:
        m = float((int(p1[1, 0]) - int(p0[1, 0])) / (int(p1[0, 0]) - int(p0[0, 0])))
    except:
        m = 10000
    print("Line slope = %f" % m)

    return m, p0, p1


def draw_fitted_line(image, fit_line):
    _, p0, p1 = calc_line_slope(image, fit_line)

    # draw the line. For my version of opencv, it wants tuples so we
    # flatten the arrays and convert
    # args: cv2.line(image, p0, p1, color, thickness)
    cv2.line(image, tuple(p0.ravel()), tuple(p1.ravel()), (255, 0, 255), 2)


def extract_rails_line(prediction):
    pred_rail_classes = [1, 2]  # the output classes that correspond to the left and right rails
    rails_lines = []
    for rail_class in pred_rail_classes:
        mask_like = np.zeros_like(prediction).astype(np.uint8)
        mask_like[np.where(prediction == rail_class)] = 255
        #mask_like[np.where(prediction == 2)] = 255

        # perform some morph operations
        kernel = np.ones((3, 3), np.uint8)
        mask_like = cv2.dilate(mask_like, kernel, iterations=1)

        # show image
        cv2.imshow('rail mask', mask_like)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # filter the 2 rail contours
        # get contours
        result = cv2.merge((mask_like, mask_like, mask_like))
        cntrs_info = []
        contours = cv2.findContours(mask_like, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        index = 0
        for cntr in contours:
            area = cv2.contourArea(cntr)
            M = cv2.moments(cntr)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cntrs_info.append((index, area, cx, cy))
            index = index + 1
            # print(index,area,cx,cy)

        # sort contours by area
        def takeSecond(elem):
            return elem[1]

        cntrs_info.sort(key=takeSecond, reverse=True)

        num_contours = 0
        for cnt in cntrs_info:
            if num_contours >= 1:
                num_contours = 0
                break
            fit_line = cv2.fitLine(contours[cnt[0]], distType=cv2.DIST_L2, param=0, reps=0.01, aeps=0.01)
            m, _, _ = calc_line_slope(result, fit_line)
            if m > -1 and m < 1:
                continue
            else:
                index = cnt[0]
                area = cnt[1]
                cx = cnt[2]
                cy = cnt[3]
                print("index:", index, "area:", area, "cx:", cx, "cy:", cy)
                cv2.drawContours(result, [contours[index]], 0, (0, 0, 255), 2)
                num_contours += 1
                # draw fitted line
                draw_fitted_line(result, fit_line)
                # store fitted line into list
                rails_lines.append(fit_line)

        # show results
        cv2.imshow("result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return rails_lines, result


def line_intersection(line1, line2):
    xdiff = (int(line1[0][0]) - int(line1[1][0]), int(line2[0][0]) - int(line2[1][0]))
    ydiff = (int(line1[0][1]) - int(line1[1][1]), int(line2[0][1]) - int(line2[1][1]))

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return int(x), int(y)


def find_D(points, Fx, KNOWN_WIDTH):

    width_pixels = math.dist(np.array(points[0]), np.array(points[1]))

    D = (Fx * KNOWN_WIDTH) / width_pixels
    print("Distance to the point is: %f" % D)

    return D

def calc_irrig_zone_width_pix(Fx, W, D):
    # irrig_zone_width = (Fx * irrig_zone_width_mm) / dist_to_camera
    irrig_zone_width = (Fx * W) / D

    return irrig_zone_width

# taken from https://math.stackexchange.com/a/175906
def extend_irrig_margins(bisector_end_point, inv_bisector_rail_point, distance_from_bisector):
    v = [inv_bisector_rail_point[0] - bisector_end_point[0], inv_bisector_rail_point[1] - bisector_end_point[1]]
    norm_v = math.sqrt(v[0] ** 2 + v[1] ** 2)
    unit_v = [v[0] / norm_v, v[1] / norm_v]

    # find extended irrigation zone points
    irrig_zone_right_point = [int(sum(x)) for x in zip(bisector_end_point, [i * distance_from_bisector for i in unit_v])] # [bisector_end_point[0], bisector_end_point[1]] + distance_from_bisector*[unit_v[0], unit_v[1]]
    irrig_zone_left_point = [int(sum(x)) for x in zip(bisector_end_point, [i * -distance_from_bisector for i in unit_v])]

    return irrig_zone_left_point, irrig_zone_right_point

def estimate_irrig_zone(pred_list, images_list=None, save_results=False):
    # generate and show colored images
    colored_imgs = generate_colored_preds(images_list, pred_list, sv_images)

    # form the masks taking only predicted rail pixels
    for i, pred in enumerate(pred_list):
        colored_imgs_opencv = np.array(colored_imgs[i])[:, :, ::-1].copy()
        # show resulting colored predicted mask
        cv2.imshow('rail pred mask', colored_imgs_opencv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # obtain rails fitted lines
        rails_lines, result_img = extract_rails_line(pred)

        # Find bisector line between the fitted contour lines (ref: https://stackoverflow.com/a/57503229/13356757)
        m1, p11, p12 = calc_line_slope(result_img, rails_lines[0])
        m2, p21, p22 = calc_line_slope(result_img, rails_lines[1])

        draw_fitted_line(colored_imgs_opencv, rails_lines[0])
        draw_fitted_line(colored_imgs_opencv, rails_lines[1])

        # point where both lines intersect
        x_fuga, y_fuga = line_intersection([p11, p12], [p21, p22])
        cv2.circle(colored_imgs_opencv, (x_fuga, y_fuga), radius=5, color=(255, 0, 0), thickness=-1)

        # calculate unit vector defined by (p11, p12) and (p21, p22)
        direction1 = find_unit_vector(p11, p12)
        direction2 = find_unit_vector(p21, p22)

        # sum both unitary vectors and calculate its unitary vector
        bisector_vector = [sum(x) for x in zip(direction1, direction2)]
        norm_bisector = math.sqrt(bisector_vector[0] ** 2 + bisector_vector[1] ** 2)
        bisector_direction = [bisector_vector[0] / norm_bisector, bisector_vector[1] / norm_bisector]
        # find its slope, independent term b and intersection with the lowest end of the image
        m_bisector = bisector_direction[1] / bisector_direction[0]
        b = y_fuga - m_bisector * x_fuga
        x_end = int((colored_imgs_opencv.shape[0] - b) / m_bisector)

        # draw bisector line
        cv2.line(colored_imgs_opencv, (x_fuga, y_fuga), (x_end, colored_imgs_opencv.shape[0]), (0, 255, 0), 2)

        # show line intersection
        cv2.imshow('lines intersection', colored_imgs_opencv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # build irrigation zone
        # inv_m_bisector is ortogonal to the bisector
        inv_m_bisector = -(1 / m_bisector)
        # inv_b is the independent term on the y = mx + inv_b line equation ortogonal to the bisector
        inv_b = colored_imgs_opencv.shape[0] - inv_m_bisector * int(p12[0][0])

        # intersection of the ortogonal bisector with rails
        x_right, y_right = line_intersection(
            [np.array([[x_end], [colored_imgs_opencv.shape[0]]]), np.array([[0], [inv_b]])], [p21, p22])
        cv2.circle(colored_imgs_opencv, (x_right, y_right), radius=5, color=(255, 0, 0), thickness=-1)
        x_left, y_left = line_intersection(
            [np.array([[x_end], [colored_imgs_opencv.shape[0]]]), np.array([[0], [inv_b]])], [p11, p12])
        cv2.circle(colored_imgs_opencv, (x_left, y_left), radius=5, color=(255, 0, 0), thickness=-1)
        # draw ortogonal bisector line
        cv2.line(colored_imgs_opencv, (x_left, y_left), (x_right, y_right), (0, 255, 255), 2)

        # show line intersection
        cv2.imshow('lines intersection', colored_imgs_opencv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # first find the distance from the central point of the bisector to the camera
        points = [[x_left, y_left], [x_right, y_right]]
        dist_rails_pixels = math.sqrt((x_left - x_right) ** 2 + (y_left - y_right) ** 2)
        dist_to_camera = find_D(points, Fx, KNOWN_DISTANCE)

        # calculate the width in pixels of the irrigation zone on the zone of the ortogonal bisector (yellow line)
        irrig_zone_width_pix = calc_irrig_zone_width_pix(Fx, WIDTH_IRRIG_ZONE, dist_to_camera)

        # extend the ortogonal bisector to the limits of the irrigation zone
        left_irrig_point, right_irrig_point = extend_irrig_margins([x_end, colored_imgs_opencv.shape[0]],
                                                                [x_right, y_right], irrig_zone_width_pix / 2)

        # find the exact point where the irrigation zone limits cut the limit of the image to form a polygon
        end_left_irrig_limit_point = line_intersection(
            [np.array([[x_fuga], [y_fuga]]), np.array([[left_irrig_point[0]], [left_irrig_point[1]]])],
            [np.array([[0], [colored_imgs_opencv.shape[0]]]),
             np.array([[colored_imgs_opencv.shape[1]], [colored_imgs_opencv.shape[0]]])])
        end_right_irrig_limit_point = line_intersection(
            [np.array([[x_fuga], [y_fuga]]), np.array([[right_irrig_point[0]], [right_irrig_point[1]]])],
            [np.array([[0], [colored_imgs_opencv.shape[0]]]),
             np.array([[colored_imgs_opencv.shape[1]], [colored_imgs_opencv.shape[0]]])])

        # draw irrigation zone limits
        cv2.line(colored_imgs_opencv, (x_fuga, y_fuga), end_left_irrig_limit_point, (0, 0, 0), 2)
        cv2.line(colored_imgs_opencv, (x_fuga, y_fuga), end_right_irrig_limit_point, (0, 0, 0), 2)

        # show line intersection
        cv2.imshow('lines intersection', colored_imgs_opencv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # write results to disk
        if save_results:
            result_folder = 'rail_extraction_postprocessing/results'
            if not os.path.exists(result_folder):
                os.mkdir(result_folder)
            cv2.imwrite(os.path.join(result_folder, os.path.basename(images_list[i])), colored_imgs_opencv)



if __name__ == '__main__':
    args = parse_args()
    images_list = glob.glob(args.r+'*'+args.t)

    # perform inference
    pred_list, sv_images = inference(args, images_list)

    # estimate the irrigation zone
    estimate_irrig_zone(pred_list, images_list=images_list, save_results=True)

    print("Finito!")
