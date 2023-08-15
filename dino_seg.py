from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
import cv2
import supervision as sv
base_model = GroundedSAM(ontology=CaptionOntology({"field": "field", "road": "road"}))
image_name = "/home/aymen/PycharmProjects/owlvit_segment_anything/dataset/test_images/6.jpeg"

mask_annotator = sv.MaskAnnotator()

image = cv2.imread(image_name)

classes = base_model.ontology.classes()

detections = base_model.predict(image_name)

labels = [
    f"{classes[class_id]} {confidence:0.2f}"
    for _, _, confidence, class_id, _
    in detections
]

annotated_frame = mask_annotator.annotate(
    scene=image.copy(),
    detections=detections
)

sv.plot_image(annotated_frame, size=(8, 8))

base_model.label(input_folder="full_images", extension="*", output_folder="outputs/")