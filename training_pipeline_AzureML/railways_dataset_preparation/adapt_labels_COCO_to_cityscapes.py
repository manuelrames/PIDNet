import json, os

def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)

data = json.loads(open('./datasets/c1ad315b-4265-4818-9017-5e6cfdba20d5.json', "r").read())

gtFine_polygons = {}

for image_info in data['images']:
    gtFine_polygons['imgHeight'] = image_info['height']
    gtFine_polygons['imgWidth'] = image_info['width']

    objects = []

    img_id = image_info['id']
    img_annot = [annot for annot in data['annotations'] if annot['image_id'] == img_id]

    for annot in img_annot:
        obj = {}
        # get only main label
        #obj['label'] = next(item for item in data['categories'] if item["id"] == annot['category_id'])['name'].split('/')[0]
        # get sublabel category
        obj['label'] = next(item for item in data['categories'] if item["id"] == annot['category_id'])['name'].split('/')[-1]
        # get polygon
        obj['polygon'] = []

        for rel_x, rel_y in pairwise(annot['segmentation'][0]):
            abs_x = round(rel_x * int(gtFine_polygons['imgWidth']))
            abs_y = round(rel_y * int(gtFine_polygons['imgHeight']))
            obj['polygon'].append([abs_x, abs_y])

        objects.append(obj)

    gtFine_polygons['objects'] = objects

    # export each json file
    output_folder = './datasets/railway_5classes_json_polygons'
    json_filepath = os.path.join(output_folder, os.path.basename(image_info['file_name'][:-4]) + '_gtFine_polygons.json')
    with open(json_filepath, 'w') as fp:
        json.dump(gtFine_polygons, fp)