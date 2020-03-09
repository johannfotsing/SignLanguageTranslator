# ECE6258_FinalProject

Authors : Lucas GRUSS, Johann FOTSING

# System requirements

We suggest using a virtual environment (you can use the Python IDE PyCharm) with the following modules loaded:
<ol>
<li> tensorflow 1.12.0</li>
<li> keras </li>
<li> numpy </li>
</ol>

# Some preparation before running the program

There are a couple of assumptions made on the environment the program is used :
<ul>
<li> The background is uniform (blank wall of any color)</li>
<li> The user is in a well lit environment but not overexposed </li>
<li> the user wears long sleeved clothing </li>
</ul>

# Procedure to follow 

## Calibrating the program

Run the script skin_calibration.py by either typing in a terminal :

    python3 skin_calibration.py+

You will see a window appear, it should be completely black. There are two slider at the bottom of the window. Increase the slider 'minval' until you can only see the outline of your face and hands. Decrease the second slider 'maxval' to remove the noise. The second slider may not change much the quality of the segmentation in your case, but depending on the background and lighting conditions, it can significantly improve the segmentation and drown out noise.

Once you are satisfied with your calibration, press the key 'c' on your keyboard to save the values you just set. If at any moment you want to quit the program, press the letter 'q'.

## Training the models

Run the script update_models.py to train the models.

## Running the main of the program 

Run the script run_me.py by either typing in a terminal :

    python3 run_me.py
    
or running it directly in an IDE.

You can try different signs from ASL. It is better if your face is shown to the camera, as the program assumes the face is present (it can actually detect your hand even when the face is not in the image, but the detection is more robust when your face is shown).
