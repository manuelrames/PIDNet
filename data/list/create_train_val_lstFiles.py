# Script that create the lst files necessary for training
import glob

images_lists = {}
images_lists['train_images_list'] = glob.glob('data/railways_night_5classes/leftImg8bit/train/*.jpg')
images_lists['val_images_list'] = glob.glob('data/railways_night_5classes/leftImg8bit/val/*.jpg')

added_string = '_gtFine_labelIds.png'

with open('data/list/railways_night_5classes/train.lst', 'w+') as the_file:
    for filename in images_lists['train_images_list']:
        line = filename.replace('data/railways_night_5classes/', '') + '\t' + filename.replace('data/railways_night_5classes/', '')[:-4].replace('leftImg8bit','gtFine') + added_string + '\n'
        the_file.write(line)

with open('data/list/railways_night_5classes/val.lst', 'w+') as the_file:
    for filename in images_lists['val_images_list']:
        line = filename.replace('data/railways_night_5classes/', '') + '\t' + filename.replace('data/railways_night_5classes/', '')[:-4].replace('leftImg8bit','gtFine') + added_string + '\n'
        the_file.write(line)