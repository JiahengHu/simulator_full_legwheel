import os
cwd = os.path.dirname(os.path.realpath(__file__))

data_subfolder = cwd

## concatentate videos for all goals
print('Concatenating videos')
vidlist = os.path.join(data_subfolder,'vidlist.txt')
# create a list of the video names for ffmpeg
fnames = ['lwllwl_live_0', 'lwllwl_live_1', 'llllll_live_2']
with open(vidlist, 'w') as f:
    for fname in fnames:

        vidpath = os.path.join(data_subfolder,fname + '.mp4')
        # ffmpeg seems to pick up the paths from the location of the file
        f.write("file '" + vidpath +"'\n")
# run concat command in ffmpeg
run_str2 = 'ffmpeg -f concat -safe 0 -i ' + vidlist + ' -c copy ' + os.path.join(data_subfolder,'output.mp4')
run_str2 = run_str2 + ' -y'
status = os.system(run_str2)
print('Status (0 is good):' + str(status))
print('Done')


cmd_status = os.system('ffmpeg -i ' + 
                  os.path.join(data_subfolder,'output.mp4') 
                  + ' -filter:v setpts=2.0*PTS ' +
                  os.path.join(data_subfolder,'output_1x.mp4') + ' -y')