"""
Treatment recommendations for each class the model can predict.

Each entry is a short, general-purpose description and a practical first
step — not a substitute for a local agricultural extension office, which
should always be the next call for a real infestation/outbreak. Keys match
the class names exactly as saved in models/class_names.json.
"""

TREATMENTS = {
    "Apple___Apple_scab": {
        "description": "Fungal disease causing olive-green to black scabby lesions on leaves and fruit.",
        "treatment": "Remove and destroy fallen leaves in autumn to reduce spore sources; apply a protectant fungicide (e.g. captan or myclobutanil) starting at bud break.",
    },
    "Apple___Black_rot": {
        "description": "Fungal disease causing purple-bordered leaf spots and rotting, mummified fruit.",
        "treatment": "Prune out dead/cankered wood and mummified fruit, then apply a labeled fungicide during the growing season.",
    },
    "Apple___Cedar_apple_rust": {
        "description": "Fungal disease needing a nearby cedar/juniper host; causes bright orange leaf spots.",
        "treatment": "Remove nearby juniper/cedar hosts if practical, and apply a preventive fungicide from pink bud stage through early summer.",
    },
    "Apple___healthy": {
        "description": "No disease detected — leaf appears healthy.",
        "treatment": "No treatment needed. Keep up routine watering, pruning, and pest monitoring.",
    },
    "Blueberry___healthy": {
        "description": "No disease detected — leaf appears healthy.",
        "treatment": "No treatment needed. Maintain acidic, well-drained soil and monitor for pests.",
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "description": "Fungal disease producing a white, powdery coating on leaves and shoots.",
        "treatment": "Improve air circulation via pruning and apply a sulfur-based or other labeled fungicide at first sign of infection.",
    },
    "Cherry_(including_sour)___healthy": {
        "description": "No disease detected — leaf appears healthy.",
        "treatment": "No treatment needed. Continue standard care and seasonal monitoring.",
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "description": "Fungal disease causing rectangular gray-to-tan lesions along leaf veins.",
        "treatment": "Rotate crops away from corn for 1-2 seasons, use resistant hybrids where possible, and apply a foliar fungicide if disease pressure is high.",
    },
    "Corn_(maize)___Common_rust_": {
        "description": "Fungal disease producing reddish-brown pustules on both leaf surfaces.",
        "treatment": "Plant rust-resistant hybrids; fungicide is rarely needed unless infection is severe and early in the season.",
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "description": "Fungal disease causing long, cigar-shaped gray-green lesions on leaves.",
        "treatment": "Use resistant hybrids and crop rotation; apply a foliar fungicide if the disease appears before tasseling.",
    },
    "Corn_(maize)___healthy": {
        "description": "No disease detected — leaf appears healthy.",
        "treatment": "No treatment needed. Continue standard field monitoring.",
    },
    "Grape___Black_rot": {
        "description": "Fungal disease causing circular brown leaf spots and shriveled, mummified fruit.",
        "treatment": "Remove mummified fruit and infected canes during dormant pruning; apply fungicide starting at bud break through fruit set.",
    },
    "Grape___Esca_(Black_Measles)": {
        "description": "Fungal trunk disease causing striped/tiger-stripe leaf discoloration and fruit spotting.",
        "treatment": "No curative fungicide exists; prune out and destroy infected wood, and avoid pruning during wet weather to limit spread.",
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "description": "Fungal disease causing irregular dark brown spots on grape leaves.",
        "treatment": "Improve canopy airflow through leaf pulling/pruning and apply a labeled fungicide during humid periods.",
    },
    "Grape___healthy": {
        "description": "No disease detected — leaf appears healthy.",
        "treatment": "No treatment needed. Continue standard vineyard care.",
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "description": "Serious bacterial disease spread by psyllid insects; causes blotchy mottled leaves and misshapen, bitter fruit.",
        "treatment": "No cure exists — remove and destroy infected trees to slow spread, and control the Asian citrus psyllid vector with approved insecticides. Contact your local agricultural extension immediately, as this disease is often reportable.",
    },
    "Peach___Bacterial_spot": {
        "description": "Bacterial disease causing small, dark, water-soaked spots on leaves and fruit.",
        "treatment": "Apply copper-based bactericide sprays during dormancy and early season; plant resistant varieties where available.",
    },
    "Peach___healthy": {
        "description": "No disease detected — leaf appears healthy.",
        "treatment": "No treatment needed. Continue standard orchard care.",
    },
    "Pepper,_bell___Bacterial_spot": {
        "description": "Bacterial disease causing small, dark, greasy-looking leaf spots that can merge and cause defoliation.",
        "treatment": "Use disease-free seed/transplants, avoid overhead watering, and apply copper-based bactericides at first sign of disease.",
    },
    "Pepper,_bell___healthy": {
        "description": "No disease detected — leaf appears healthy.",
        "treatment": "No treatment needed. Continue standard care and monitoring.",
    },
    "Potato___Early_blight": {
        "description": "Fungal disease causing dark, concentric-ring ('target') spots on lower leaves first.",
        "treatment": "Rotate crops, remove infected debris, and apply a labeled fungicide at first symptoms, especially in warm, humid conditions.",
    },
    "Potato___Late_blight": {
        "description": "Aggressive water-mold disease causing dark, water-soaked lesions that spread rapidly — the cause of the historic Irish potato famine.",
        "treatment": "Act quickly: remove and destroy infected plants, avoid overhead irrigation, and apply a protectant fungicide preventively during cool, wet weather.",
    },
    "Potato___healthy": {
        "description": "No disease detected — leaf appears healthy.",
        "treatment": "No treatment needed. Continue standard care and monitoring.",
    },
    "Raspberry___healthy": {
        "description": "No disease detected — leaf appears healthy.",
        "treatment": "No treatment needed. Continue standard care and monitoring.",
    },
    "Soybean___healthy": {
        "description": "No disease detected — leaf appears healthy.",
        "treatment": "No treatment needed. Continue standard field monitoring.",
    },
    "Squash___Powdery_mildew": {
        "description": "Fungal disease producing a white, powdery coating on leaves and stems.",
        "treatment": "Improve air circulation, water at the base rather than overhead, and apply a sulfur or potassium bicarbonate fungicide at first sign.",
    },
    "Strawberry___Leaf_scorch": {
        "description": "Fungal disease causing small purple spots that merge into larger scorched-looking blotches.",
        "treatment": "Remove infected leaves after harvest, avoid overhead watering, and apply a labeled fungicide if pressure is high.",
    },
    "Strawberry___healthy": {
        "description": "No disease detected — leaf appears healthy.",
        "treatment": "No treatment needed. Continue standard care and monitoring.",
    },
    "Tomato___Bacterial_spot": {
        "description": "Bacterial disease causing small, dark, greasy spots on leaves, stems, and fruit.",
        "treatment": "Use disease-free seed, avoid overhead watering and working plants when wet, and apply copper-based bactericides early.",
    },
    "Tomato___Early_blight": {
        "description": "Fungal disease causing dark, concentric-ring spots starting on older/lower leaves.",
        "treatment": "Remove infected lower leaves, mulch to reduce soil splash, and apply a labeled fungicide at first symptoms.",
    },
    "Tomato___Late_blight": {
        "description": "Aggressive water-mold disease causing large, dark, water-soaked lesions that spread quickly in cool, wet weather.",
        "treatment": "Remove and destroy infected plants promptly, avoid overhead irrigation, and apply a protectant fungicide preventively if the disease is active in your area.",
    },
    "Tomato___Leaf_Mold": {
        "description": "Fungal disease causing pale spots on upper leaf surfaces and olive-green mold underneath — common in humid greenhouses.",
        "treatment": "Improve ventilation and reduce humidity, avoid wetting foliage, and apply a labeled fungicide if needed.",
    },
    "Tomato___Septoria_leaf_spot": {
        "description": "Fungal disease causing many small, circular spots with dark borders and light centers.",
        "treatment": "Remove infected lower leaves, mulch to reduce soil splash, and apply a labeled fungicide at first symptoms.",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "description": "Tiny mite pest causing stippled, yellowing leaves and fine webbing, especially in hot, dry conditions.",
        "treatment": "Rinse leaves (especially undersides) with water, introduce predatory mites, or apply insecticidal soap/miticide if infestation is heavy.",
    },
    "Tomato___Target_Spot": {
        "description": "Fungal disease causing brown, concentric-ring spots on leaves, stems, and fruit.",
        "treatment": "Improve air circulation, remove infected debris, and apply a labeled fungicide at first symptoms.",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "description": "Viral disease spread by whiteflies, causing upward-curling, yellowing leaves and stunted growth.",
        "treatment": "No cure exists — remove and destroy infected plants, and control the whitefly vector with insecticides or reflective mulch to prevent spread.",
    },
    "Tomato___Tomato_mosaic_virus": {
        "description": "Viral disease causing mottled light/dark green leaf patterns and stunted, distorted growth.",
        "treatment": "No cure exists — remove and destroy infected plants, disinfect tools between plants, and avoid handling plants after using tobacco products (the virus can spread this way).",
    },
    "Tomato___healthy": {
        "description": "No disease detected — leaf appears healthy.",
        "treatment": "No treatment needed. Continue standard care and monitoring.",
    },
}


def get_treatment(class_name: str) -> dict:
    """Look up treatment info for a predicted class, with a safe fallback."""
    return TREATMENTS.get(
        class_name,
        {
            "description": "No information available for this class.",
            "treatment": "Consult a local agricultural extension office for guidance.",
        },
    )
