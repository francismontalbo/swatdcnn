"""
This program/model was made by Francis Jesmar P. Montalbo for the publication article entitled
"Automated Diagnosis of Diverse Coffee Leaf Images through a Stage-Wise Aggregated Triple Deep Convolutional Neural Network"

You can use this as you please. But please do consider citing my work and the data sources. Thank you very much!
Any further assistance needed, please e-mail me at francismontalbo@ieee.org or francismontalbo@g.batstate-u.edu.ph

Please refrain from changing anything in the code performing simulations.

You may visit my official webpage at https://francismontalbo.github.io

"""
print("INITIALIZING SWAT-DCNN MODEL")
print("THIS MAY TAKE A WHILE , PLEASE BE PATIENT")

import os
import logging

import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
from colored import fg, bg, attr

import efficientnet.keras as efn 
from efficientnet.keras import preprocess_input

import keras
from keras import backend as K
from keras.models import Model, Input, load_model
from keras.preprocessing.image import image
from keras import applications

# from keras.preprocessing.image import ImageDataGenerator
# import itertools



tf.get_logger().setLevel(logging.ERROR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

font = cv2.FONT_HERSHEY_SIMPLEX

#Dictionary

UNHEALTHY_CLR = 0
UNHEALTHY_BSL = 1
UNHEALTHY_SM = 2

SWATDCNN_RESULT_TO_UNHEALTHY = {
    UNHEALTHY_CLR: "Coffee Leaf Rust (CLR)",
    UNHEALTHY_BSL: "Brown Spot Lesions (BSL)",
    UNHEALTHY_SM: "Sooty Molds (SM)"
}

BSL_CLS = 0
BSL_PLS = 1
BSL_CLM = 2
BSL_RSM = 3

SWATDCNN_RESULT_TO_BSL = {
    BSL_CLS: "Cercospora Leaf Spots (CLS)",
    BSL_PLS: "Phoma Leaf Spots (PLS)",
    BSL_CLM: "Coffee Leaf Miner (CLM)",
    BSL_RSM: "Red Spider Mite (RSM)"
}

def load_models():
    """
    Loads the models.

    Make sure to train the models using the Jupyter-notebook files 
    or use the pre-trained weights for a faster and easier simulation.

    Assign the appropriate folder routes.

    """
    
    swatdcnn_stage_1 = load_model("weights/tdcnn/T_DCNN_Stage-1.h5") #EfficientNetB0 + DenseNet121 + VGG16 trained for 2 classes
    print()
    print("T-DCNN Backbone stage 1 loading complete")
    print()
    swatdcnn_stage_2 = load_model("weights/tdcnn/T_DCNN_Stage-2.h5") #EfficientNetB0 + DenseNet121 + VGG16 trained for 3 classes
    print("T-DCNN Backbone stage 2 loading complete")
    print()
    swatdcnn_stage_3 = load_model("weights/tdcnn/T_DCNN_Stage-3.h5") #EfficientNetB0 + DenseNet121 + VGG16 trained for 4 classes
    print("T-DCNN Backbone stage 3 loading complete")
    print()

    print("SWAT-DCNN INITIALIZED!")
    print()

    return swatdcnn_stage_1, swatdcnn_stage_2, swatdcnn_stage_3

#################################################################################################

def swatdcnn_stage1(preprocessed_image, model):
    swatdcnn_stage_1_preds = model.predict(preprocessed_image)
    swatdcnn_stage_1_result = np.argmax(swatdcnn_stage_1_preds[0])
    swatdcnn_stage_1_conf = max(100 * swatdcnn_stage_1_preds[0])
    swatdcnn_stage_1_is_unhealthy = swatdcnn_stage_1_result != 0

    return {
        "is_unhealthy": swatdcnn_stage_1_is_unhealthy,
        "confidence_level": swatdcnn_stage_1_conf,
    }

def swatdcnn_stage2(preprocessed_image, model):
    swatdcnn_stage_2_preds = model.predict(preprocessed_image)
    swatdcnn_stage_2_result = np.argmax(swatdcnn_stage_2_preds[0])
    swatdcnn_stage_2_conf = max(100 * swatdcnn_stage_2_preds[0])

    return {
        "disease": swatdcnn_stage_2_result,
        "confidence_level": swatdcnn_stage_2_conf,
    }

def swatdcnn_stage3(preprocessed_image, model):
    swatdcnn_stage_3_preds = model.predict(preprocessed_image)
    swatdcnn_stage_3_result = np.argmax(swatdcnn_stage_3_preds[0])
    swatdcnn_stage_3_conf = max(100 * swatdcnn_stage_3_preds[0])

    return {
        "brownspot": swatdcnn_stage_3_result,
        "confidence_level": swatdcnn_stage_3_conf,
    }

def swat_dcnn(input_image, swatdcnn_stage_1, swatdcnn_stage_2, swatdcnn_stage_3):
    # Load the image.
    img1 = image.load_img(input_image)
    plt.imshow(img1)

    img = image.load_img(input_image, target_size=(224, 224))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = efn.preprocess_input(x)

    # CONDITIONS
    #################################################################################################
    # SWAT-DCNN (Stage-1)
    swatdcnn_stage_1_result = swatdcnn_stage1(x, swatdcnn_stage_1)

    #################################################################################################
    # SWAT-DCNN (Stage-2)
    # Run iff Stage-1 result is unhealthy.
    if swatdcnn_stage_1_result["is_unhealthy"]:
        swatdcnn_stage_2_result = swatdcnn_stage2(x, swatdcnn_stage_2)
    else:
        swatdcnn_stage_2_result = None

    #################################################################################################
    # SWAT-DCNN (Stage-3)
    # Run iff Stage-2 result is `UNHEALTHY_BSL`.
    if swatdcnn_stage_2_result and swatdcnn_stage_2_result["disease"] == UNHEALTHY_BSL:
        swatdcnn_stage_3_result = swatdcnn_stage3(x, swatdcnn_stage_3)
    else:
        swatdcnn_stage_3_result = None

    return {
        "swatdcnn_stage1": swatdcnn_stage_1_result,
        "swatdcnn_stage2": swatdcnn_stage_2_result,
        "swatdcnn_stage3": swatdcnn_stage_3_result
    }

#################################################################################################
# SWAT-DCNN (Stage-1) RESULTS REPORT

def print_report(results):
    print("SWAT-DCNN (Stage-1) Results")
    print("======================")

    if results["swatdcnn_stage1"]["is_unhealthy"]:
        print("SWAT-DCNN Stage-1 classified the image as a %sunhealthy%s coffee leaf." % (fg("red"), attr("reset")))
    else:
        print("SWAT-DCNN classified the image as a %shealthy%s coffee leaf."% (fg("green"), attr("reset")))

    print(
        "The SWAT-DCNN model confidence level is %(fg)s%(confidence)s%%%(reset)s"
        % {
            "fg": fg("yellow"),
            "confidence": results["swatdcnn_stage1"]["confidence_level"],
            "reset": attr("reset"),
        }
    )

    print()

    #################################################################################################
    # SWAT-DCNN (Stage-2) RESULTS REPORT

    if results["swatdcnn_stage2"]:
        print("SWAT-DCNN (Stage-2) Results")
        print("======================")
        print("SWAT-DCNN Stage-2 classified the disease as %(fg)s%(disease)s%(reset)s." % {
            "fg": fg("red"),
            "disease": SWATDCNN_RESULT_TO_UNHEALTHY[results["swatdcnn_stage2"]["disease"]],
            "reset": attr("reset"),
        })

        print(
            "The SWAT-DCNN model confidence level is %(fg)s%(confidence)s%%%(reset)s"
            % {
                "fg": fg("yellow"),
                "confidence": results["swatdcnn_stage2"]["confidence_level"],
                "reset": attr("reset"),
            }
        )

        print()

    #################################################################################################
    # SWAT-DCNN (Stage-3) RESULTS REPORT

    if results["swatdcnn_stage3"]:
        print("SWAT-DCNN (Stage-3) Results")
        print("======================")
        print("SWAT-DCNN Stage-3 classified the brownspot as %(fg)s%(brownspot)s%(reset)s." % {
            "fg": fg("red"),
            "brownspot": SWATDCNN_RESULT_TO_BSL[results["swatdcnn_stage3"]["brownspot"]],
            "reset": attr("reset"),
        })

        print(
            "The SWAT-DCNN model confidence level is %(fg)s%(confidence)s%%%(reset)s"
            % {
                "fg": fg("yellow"),
                "confidence": results["swatdcnn_stage3"]["confidence_level"],
                "reset": attr("reset"),
            }
        )

        print()


def show_gui(input_image, results):
    if results["swatdcnn_stage3"]:
        label = SWATDCNN_RESULT_TO_BSL[results["swatdcnn_stage3"]["brownspot"]]
        label = "{}: {: .5f}%".format(label, results["swatdcnn_stage3"]["confidence_level"])
    elif results["swatdcnn_stage2"]:
        label = SWATDCNN_RESULT_TO_UNHEALTHY[results["swatdcnn_stage2"]["disease"]]
        label = "{}: {: .5f}%".format(label, results["swatdcnn_stage2"]["confidence_level"])
    else:
        label = "healthy: {:.5f}%".format(results["swatdcnn_stage1"]["confidence_level"])

    display_img = cv2.imread(input_image)
    img_size = cv2.resize(display_img, (1024, 1024))
    cv2.putText(img_size, label, (10, 100), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Test sample", img_size)
    cv2.waitKey(0)

#################################################################################################
# SYSTEM USER PROMPT ONLY*
# Test any image in your test folder, just the filename will do.
# e.g.: test-3.jpg
if __name__ == "__main__":
    test_folder = "test/"

    swatdcnn_stage_1, swatdcnn_stage_2, swatdcnn_stage_3 = load_models()

    while True:
        while True:
            try:
                input_image = input(
                    "Please enter a file from your test folder with the coffee leaf images ex: ('test-2.jpg'): "
                )

                result = swat_dcnn(test_folder + input_image, swatdcnn_stage_1, swatdcnn_stage_2, swatdcnn_stage_3)
            except IOError:
                print(
                    "The image was not found or is invalid. Please try again, or press CTRL+C to exit the program..."
                )
            else:
                break

        print_report(result)

        print("Close the image window to continue. . .")
        show_gui(test_folder + input_image, result)
