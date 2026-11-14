# Decisions log (one line each: date - decision - why)

- 2026-11-14 - IFC room areas: authored quantity first, geometry footprint area as fallback - Esplanades model has no authored areas; fallback takes IFC area eval from 550/580 to 580/580.
- 2026-11-14 - Floor-plan geometry uses shared room/icon segmentation with QA counts as an auxiliary loss; image-level answers alone cannot provide object locations.
- 2026-11-14 - Swiss dwellings answered from the DXF twin, not the PDF/PNG - DXF has semantic layers (DOOR, WINDOW, ROOM_OUTLINE, ROOM_TEXT, FIXTURE, UNIT_OUTLINE); geometry gives exact counts/areas with entity handles as evidence.
- 2026-11-14 - Fixture rules: rounded-rect basin >=1.2 m = bathtub; elliptical bowl with drain circle = washbasin, without = toilet; rectangular tray with drain = shower - consistent drawing convention, 1614/1614 counts on preview.
- 2026-11-14 - Unnamed numbered rooms are included in apartment/floor totals - source room type is not drawn; accounts for the remaining 20/744 area misses.
- 2026-11-14 - CubiCasa room-type questions answered by OCR of Finnish room labels (RapidOCR, 4 orientations, merged) - preview: bedrooms 70%, balconies 88%, WC rooms 69%, saunas 64%, storeys 84%; kitchens/bathrooms answer the typical value 1 (95%/88%) because OCR was worse than that; flagged review_recommended.
- 2026-11-14 - Uploaded-file parsing and cache are scoped to a conversation instance and lazy-loaded; this avoids persistent storage while isolating chats.
- 2026-11-14 - Chat API keeps conversation state in process and on local temporary storage; use one server worker and close sessions to clear files - avoids introducing a database before persistence is needed.
- 2026-11-14 - Prefer official CubiCasa5K pretrained segmentation for floor-plan counts when its checkout and checkpoint are configured; skip project-specific CNN training and SVG mask generation for that path.
- 2026-11-14 - Allow users to override uncertain raster drawing classification per uploaded file; automatic image classifiers can abstain and must not block the selected specialist parser.
- 2026-11-14 - IFC counts/quantities in ifc_qa.py: elements assigned to storeys by walking containment/aggregation; a stair is an IfcStair not part of another IfcStair; quantities from authored Qto scaled by project units, geometry volume as fallback - preview ifc_models 1023/1049, all remaining misses need an IFC not in the preview (FZK-Haus) or one room label.
- 2026-11-14 - Non-floor-plan rasters share RapidOCR line extraction for title-block fields, text lookups, and printed dimensions; do not infer symbol counts or scaled geometry without a dedicated detector or calibration.
- 2026-11-14 - IFC text questions use authored IFC header, spatial, type, dimensional and property data; return not_found when required quantities/properties are absent rather than estimate.
- 2026-11-14 - Raster-only PDF pages with fewer than five embedded words and at least one image use shared RapidOCR; OCR coordinates are scaled to page coordinates and symbol counts remain unsupported.
- 2026-11-14 - DWG count/area questions use ezdxf ODA conversion followed by the existing layer-based CAD answerer; files without supported semantic layers remain unanswerable.
