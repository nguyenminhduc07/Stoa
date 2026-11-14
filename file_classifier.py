from builtins import FileNotFoundError
from importlib import import_module
from pathlib import Path
from typing import Dict, Optional

class DocumentRouter:
    # CLIP provides useful zero-shot classification without training a model on
    # this dataset. It is loaded only if an image's source does not determine
    # its drawing type.
    # Base-Patch32 is substantially lighter than Large-Patch14 while still
    # providing useful zero-shot routing for uploaded architectural drawings.
    IMAGE_MODEL = "openai/clip-vit-base-patch32"
    IMAGE_LABELS = [
        "a technical architectural floor plan drawing from directly above, showing room layouts, walls, doors and windows",
        "a colored residential or real-estate floor plan diagram from above showing rooms, walls, doors, windows and furniture",
        "a technical exterior elevation drawing showing the facade of a building",
        "a technical building section drawing cut vertically through a building",
        "a close-up construction detail drawing of a building component or assembly",
        "a technical site plan showing buildings and features on a plot of land",
        "a map or landscape plan showing roads, terrain or outdoor space",
        "a photograph or non-technical image, not an architectural drawing",
    ]
    LABEL_TO_TYPE = {
        IMAGE_LABELS[0]: "floor_plan",
        IMAGE_LABELS[1]: "floor_plan",
        IMAGE_LABELS[2]: "elevation",
        IMAGE_LABELS[3]: "section",
        IMAGE_LABELS[4]: "detail",
        IMAGE_LABELS[5]: "site_plan",
        IMAGE_LABELS[6]: "map",
        IMAGE_LABELS[7]: "other",
    }

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        
        # Define extension routing rules
        self.bim_extensions = {'.ifc', '.rvt'}
        self.vector_extensions = {'.pdf', '.dwg', '.dxf'}
        self.image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
        self._image_classifier = None
        
        # Pipeline output schema
        self.routed_files = {
            "bim_models": [],
            "vector_files": [],
            "images": {
                "floor_plans": [],
                "elevations": [],
                "sections": [],
                "unsupported": []
            },
            "unrecognized": []
        }

    def run_pipeline(self) -> Dict:
        """Traverses the dataset and routes all files."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path {self.dataset_path} does not exist.")

        for file_path in self.dataset_path.rglob('*'):
            if file_path.is_file():
                self._route_file(file_path)
                
        return self.routed_files

    def _route_file(self, file_path: Path):
        """Routes a single file based on extension and ML classification."""
        ext = file_path.suffix.lower()

        if ext in self.bim_extensions:
            self.routed_files["bim_models"].append(str(file_path))
            
        elif ext in self.vector_extensions:
            self.routed_files["vector_files"].append(str(file_path))
            
        elif ext in self.image_extensions:
            # Trigger the ML classifier for raster images
            drawing_type = self._classify_image(file_path)
            
            # Map the prediction to the schema
            if drawing_type == "floor_plan":
                self.routed_files["images"]["floor_plans"].append(str(file_path))
            elif drawing_type == "elevation":
                self.routed_files["images"]["elevations"].append(str(file_path))
            elif drawing_type == "section":
                self.routed_files["images"]["sections"].append(str(file_path))
            else:
                self.routed_files["images"]["unsupported"].append(str(file_path))
                
        else:
            self.routed_files["unrecognized"].append(str(file_path))

    def _classify_image(self, file_path: Path) -> str:
        """Classify a raster drawing, using reliable dataset provenance first.

        Returns a drawing type, or ``uncertain`` when the model cannot make a
        sufficiently clear choice. Install ``transformers``, ``torch`` and
        ``Pillow`` to enable zero-shot classification for mixed-source images.
        The first model use downloads the weights if they are not cached.
        """
        source = next((part.lower() for part in file_path.parts
                       if part.lower() in {"cubicasa5k", "swiss_dwellings"}), None)
        if source:
            return "floor_plan"

        try:
            from PIL import Image, UnidentifiedImageError
            with Image.open(file_path) as image:
                image = image.convert("RGB")
        except ImportError as exc:
            raise RuntimeError(
                "Classifying mixed-source raster images needs Pillow, transformers, and torch. "
                "Install them with: pip install pillow transformers torch"
            ) from exc
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"Cannot read image file: {file_path}") from exc

        if self._image_classifier is None:
            try:
                pipeline = import_module("transformers").pipeline
            except ImportError as exc:
                raise RuntimeError(
                    "Image classification needs transformers, torch, and Pillow. "
                    "Install them with: pip install transformers torch pillow"
                ) from exc
            self._image_classifier = pipeline(
                task="zero-shot-image-classification",
                model=self.IMAGE_MODEL,
            )

        predictions = self._image_classifier(image, candidate_labels=self.IMAGE_LABELS)
        if not predictions:
            return "uncertain"

        scores_by_type: Dict[str, float] = {}
        for item in predictions:
            dtype = self.LABEL_TO_TYPE.get(item["label"], "uncertain")
            scores_by_type[dtype] = scores_by_type.get(dtype, 0.0) + float(item["score"])
        ranked = sorted(scores_by_type.items(), key=lambda kv: kv[1], reverse=True)
        best_type, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        if best_score >= 0.24 and (best_score - second_score) >= 0.03:
            return best_type
        try:
            from image_parser import is_likely_floorplan_by_structure
            if is_likely_floorplan_by_structure(file_path):
                return "floor_plan"
        except Exception:
            pass
        return "uncertain"


# --- Execution Example ---
if __name__ == "__main__":
    # Point this to your root dataset folder containing all the mixed sources
    DATASET_DIRECTORY = "./construction_data"
    
    # Initialize and run
    router = DocumentRouter(DATASET_DIRECTORY)
    
    try:
        results = router.run_pipeline()
        
        # Output results to pass to downstream parsers
        print(f"Found {len(results['bim_models'])} BIM models for IfcOpenShell parsing.")
        print(f"Found {len(results['vector_files'])} Vector files for PDF/DXF parsing.")
        print(f"Found {len(results['images']['floor_plans'])} Floor Plans for Vision Extraction.")
        print(f"Ignored {len(results['images']['elevations'])} Elevations.")
        
    except FileNotFoundError as e:
        print(e)
