# Decisions log (one line each: date - decision - why)

- 2026-11-14 - IFC room areas: authored quantity first, geometry footprint area as fallback - Esplanades model has no authored areas; fallback takes IFC area eval from 550/580 to 580/580.
- 2026-11-14 - Floor-plan geometry uses shared room/icon segmentation with QA counts as an auxiliary loss; image-level answers alone cannot provide object locations.
- 2026-11-14 - Swiss dwellings answered from the DXF twin, not the PDF/PNG - DXF has semantic layers (DOOR, WINDOW, ROOM_OUTLINE, ROOM_TEXT, FIXTURE, UNIT_OUTLINE); geometry gives exact counts/areas with entity handles as evidence.
- 2026-11-14 - Fixture rules: rounded-rect basin >=1.2 m = bathtub; elliptical bowl with drain circle = washbasin, without = toilet; rectangular tray with drain = shower - consistent drawing convention, 1614/1614 counts on preview.
- 2026-11-14 - Unnamed numbered rooms are included in apartment/floor totals - source room type is not drawn; accounts for the remaining 20/744 area misses.
- 2026-11-14 - Uploaded-file parsing and cache are scoped to a conversation instance and lazy-loaded; this avoids persistent storage while isolating chats.
- 2026-11-14 - Chat API keeps conversation state in process and on local temporary storage; use one server worker and close sessions to clear files - avoids introducing a database before persistence is needed.
- 2026-11-14 - Prefer official CubiCasa5K pretrained segmentation for floor-plan counts when its checkout and checkpoint are configured; skip project-specific CNN training and SVG mask generation for that path.
