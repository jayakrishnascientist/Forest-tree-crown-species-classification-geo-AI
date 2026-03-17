# QGIS + QField Workflow for Tree Crown Ground-Truthing

## Overview
This project provides a structured workflow for ground-truthing tree crown polygons using QGIS and QField. It integrates remote sensing outputs (e.g., Detectree2) with field-based validation to create accurate, labeled ecological datasets.

## Workflow Summary
1. Data Preparation  
2. QGIS Project Setup  
3. Attribute Configuration  
4. Visualization  
5. QField Cloud Deployment  
6. Field Data Collection  
7. Data Synchronization  

## 1. Data Preparation

### Steps
- Compress orthomosaic (OM) raster without altering georeferencing  
- Convert crown polygons from .geojson to .gpkg  
- Ensure CRS consistency across all datasets  

Efficient field workflows require lightweight datasets. Compression reduces file size while preserving spatial accuracy. GeoPackage (.gpkg) is preferred because it is faster, more stable, and fully supported in mobile environments like QField.

## 2. QGIS Project Setup

### Steps
- Create a new QGIS project  
- Set project CRS  
- Load orthomosaic (OM) and crown polygons (GPKG)  
- Verify spatial alignment  

QGIS acts as the central environment where spatial layers are integrated. Correct CRS alignment ensures that all datasets overlay accurately, which is critical for reliable field validation.

## 3. Attribute Schema Configuration

### Steps
Add fields to crown polygon layer:

- crown_id (Integer)  
- species (String)  
- description (String)  
- photo (String)  
- status (String)  
- tree_type (String)  
- health (String)  

### Form Customization
Set widget types in Attribute Form:

- species → Value Map (species list)  
- status → pending / completed  
- tree_type → acacia / non-acacia  
- health → flowering / leaf shed / full canopy / snag-dead  
- photo → Attachment  

Standardized attribute forms reduce human error during field data collection. Value maps enforce controlled vocabularies, ensuring consistent and analyzable datasets.

## 4. Visualization and Styling

### Steps
- Set polygon fill to transparent  
- Enable boundary outline  
- Apply rule-based styling:
  - Completed → Green  
  - Pending → Red  

Visual cues improve field efficiency. Color-coded status helps users quickly identify which trees are already labeled and which require attention.

## 5. QField Cloud Deployment

### Steps
- Create QField Cloud account  
- Install QFieldCloud plugin in QGIS  
- Upload project to cloud  
- Ensure all paths are relative  

Cloud synchronization enables seamless data transfer between desktop (QGIS) and mobile (QField), supporting collaborative and real-time workflows.

## 6. Field Data Collection (QField)

### Steps
- Install QField app  
- Login and download project  
- Use GPS to locate crowns  
- Update attributes (species, status, health, tree_type)  
- Capture photos  
- Save edits  

QField enables in-situ data validation. Integrating GPS with spatial layers allows precise mapping of ecological attributes directly in the field.

## 7. Collaboration

### Steps
- Add contributors in QField Cloud  
- Assign roles (Editor)  
- Share project access  

Collaborative editing allows multiple users to collect data simultaneously, significantly speeding up large-scale ecological surveys.

## 8. Data Synchronization

### Steps
- Push updates from QField to Cloud  
- Sync project back to QGIS  
- Or download updated GPKG  

Bidirectional synchronization ensures that field updates are reflected in the master dataset, enabling continuous refinement and analysis.

## Best Practices

- Always maintain CRS consistency  
- Use GeoPackage instead of GeoJSON  
- Keep backups before syncing  
- Use default values (e.g., status = pending)  
- Optimize raster size for mobile performance  

## Output

- Fully labeled tree crown dataset  
- Georeferenced images linked to crowns  
- Ready for ecological analysis, machine learning training, and reporting

## Use Case

This workflow is suitable for forest monitoring, species classification, UAV-based ecological surveys, and ground-truth data collection.
