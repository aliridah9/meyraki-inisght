import io
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw
import urllib.request
from typing import Dict, Any, List, Optional, Tuple
import scipy.stats
import matplotlib.cm
import httpx
import json
import time
import uuid
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage

from app.services.cloudinary_service import cloudinary_service
from app.services.project_service import project_service
from app.models.project import Project, ObjectiveType, SpaceType


class AnalysisService:
    @staticmethod
    async def process_floorplan(image_url: str) -> Dict[str, Any]:
        """
        Process a floorplan image using OpenCV
        
        Args:
            image_url: URL of the floorplan image
            
        Returns:
            Processed floorplan data
        """
        try:
            # Download the image
            response = urllib.request.urlopen(image_url)
            image_data = response.read()
            
            # Convert to OpenCV format
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                raise Exception("Failed to decode image. Ensure it's a valid image format.")

            # Image Scaling
            TARGET_WIDTH = 1024.0
            original_height, original_width = img.shape[:2]
            
            if original_width == 0:
                raise Exception("Original image width is zero, cannot scale.")

            ratio = TARGET_WIDTH / original_width
            target_height = int(original_height * ratio)
            
            # Ensure target_height is positive
            if target_height <= 0:
                target_height = 1 # Set to a minimum positive value

            img_scaled = cv2.resize(img, (int(TARGET_WIDTH), target_height), interpolation=cv2.INTER_AREA)
            
            # Convert the scaled image to grayscale
            gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)

            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 11, 2)

            # Find contours using the new thresholded image
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Get major zones based on contours
            zones = []
            for i, contour in enumerate(contours):
                # Only consider large enough areas (filter out noise)
                if cv2.contourArea(contour) > 500:
                    x, y, w, h = cv2.boundingRect(contour)
                    zones.append({
                        "id": f"zone_{i}",
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h),
                        "area": int(cv2.contourArea(contour))
                    })
            
            # Calculate total floor area using scaled image dimensions
            height, width = img_scaled.shape[:2]
            total_area = width * height
            
            # Calculate space efficiency
            occupied_area = sum(zone["area"] for zone in zones)
            efficiency = occupied_area / total_area if total_area > 0 else 0
            
            return {
                "width": width, # Return scaled width
                "height": height, # Return scaled height
                "zones": zones,
                "total_area": total_area,
                "occupied_area": occupied_area,
                "efficiency": efficiency
            }
            
        except Exception as e:
            raise Exception(f"Failed to process floorplan: {str(e)}")
    
    @staticmethod
    async def process_usage_data(csv_url: str) -> Dict[str, Any]:
        """
        Process and analyze CSV usage data
        
        Args:
            csv_url: URL of the CSV file
            
        Returns:
            Processed usage data
        """
        try:
            # Download the CSV
            async with httpx.AsyncClient() as client:
                response = await client.get(csv_url)
                csv_data = response.content
            
            # Parse CSV
            df = pd.read_csv(io.BytesIO(csv_data))
            
            # Basic stats
            row_count = len(df)
            column_count = len(df.columns)
            
            # Check if we have x, y coordinates
            has_coordinates = "x" in df.columns and "y" in df.columns
            
            # Get hotspots (areas with most user activity)
            hotspots = []
            
            if has_coordinates:
                # Use K-means to find clusters of activity
                coords = df[["x", "y"]].values
                # Scale the coordinates
                scaler = StandardScaler()
                scaled_coords = scaler.fit_transform(coords)
                
                # Determine optimal number of clusters (3-5 for MVP)
                num_clusters = min(5, len(coords) // 10) if len(coords) > 30 else 3
                
                # Apply K-means
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                df["cluster"] = kmeans.fit_predict(scaled_coords)
                
                # Get cluster centers
                centers = scaler.inverse_transform(kmeans.cluster_centers_)
                
                # Count points in each cluster
                cluster_counts = df["cluster"].value_counts().to_dict()
                
                # Create hotspots
                for i, center in enumerate(centers):
                    count = cluster_counts.get(i, 0)
                    hotspots.append({
                        "id": f"hotspot_{i}",
                        "x": int(center[0]),
                        "y": int(center[1]),
                        "strength": count / row_count if row_count > 0 else 0,
                        "count": int(count)
                    })
            
            result_data = {
                "row_count": row_count,
                "column_count": column_count,
                "has_coordinates": has_coordinates,
                "hotspots": hotspots,
                "columns": list(df.columns)
            }

            if has_coordinates:
                result_data["points"] = df[["x", "y"]].values.tolist()

            return result_data
            
        except Exception as e:
            raise Exception(f"Failed to process usage data: {str(e)}")
    
    @staticmethod
    async def generate_heatmap(floorplan_data: Dict[str, Any], usage_data: Dict[str, Any], 
                          floorplan_url: str) -> Dict[str, Any]:
        """
        Generate a heatmap overlay based on floorplan and usage data
        
        Args:
            floorplan_data: Processed floorplan data
            usage_data: Processed usage data
            floorplan_url: URL of the original floorplan
            
        Returns:
            Heatmap data including Cloudinary URL
        """
        try:
            # Download the floorplan
            response = urllib.request.urlopen(floorplan_url)
            image_data = response.read()
            
            # Convert to Image
            base_floorplan_pil = Image.open(io.BytesIO(image_data))
            width, height = base_floorplan_pil.size

            final_composite_pil = base_floorplan_pil.convert('RGBA')

            # KDE Layer
            usage_points = usage_data.get("points")
            if usage_points and len(usage_points) >= 3: # Need a few points for KDE
                data = np.array(usage_points).T
                try:
                    # Check for issues that might cause KDE to fail
                    if data.shape[0] < 2: # Need at least 2 dimensions
                        raise ValueError("Data for KDE must have at least 2 dimensions.")
                    if data.shape[1] < 2: # Need at least 2 points for meaningful KDE
                         raise ValueError("Not enough data points for a meaningful KDE.")
                    
                    # Check for collinearity or too few points for reliable KDE
                    # A simple check for condition number; might need more robust checks
                    if np.linalg.cond(np.cov(data)) > 1e8 : # Condition number too high
                         pass # Potentially skip KDE or log a warning, or use simpler heatmap

                    kde = scipy.stats.gaussian_kde(data)

                    # Create grid, ensuring division by non-zero
                    grid_width_steps = int(width / (width/100.0)) if width > 0 else 100
                    grid_height_steps = int(height / (height/100.0)) if height > 0 else 100

                    # Ensure steps are not zero
                    grid_width_steps = max(1, grid_width_steps)
                    grid_height_steps = max(1, grid_height_steps)

                    xx, yy = np.mgrid[0:width:complex(0, grid_width_steps), 0:height:complex(0, grid_height_steps)]

                    density_map = kde(np.vstack([xx.ravel(), yy.ravel()]))
                    density_map_reshaped = density_map.reshape(xx.shape).T # Transpose to match image coords

                    # Normalize density_map_reshaped to 0-1
                    min_density = np.min(density_map_reshaped)
                    max_density = np.max(density_map_reshaped)
                    if max_density == min_density:
                        density_norm = np.zeros_like(density_map_reshaped)
                    else:
                        density_norm = (density_map_reshaped - min_density) / (max_density - min_density)

                    # Create an empty RGBA overlay for KDE
                    # Note: density_norm is (grid_height_steps, grid_width_steps)
                    # We need to map this onto the full image size (height, width)
                    # This means each cell in density_norm covers multiple pixels if grid_steps < image_dims
                    # For simplicity here, we'll create a full size overlay and assign colors based on nearest grid point
                    # A more accurate way would be to interpolate or resize density_norm to (height, width)

                    # Resize density_norm to full image dimensions using PIL
                    density_norm_pil = Image.fromarray((density_norm * 255).astype(np.uint8))
                    density_norm_resized_pil = density_norm_pil.resize((width, height), Image.Resampling.LANCZOS)
                    density_norm_resized = np.array(density_norm_resized_pil) / 255.0

                    kde_overlay_rgba_np = np.zeros((height, width, 4), dtype=np.uint8)

                    for r_idx in range(height):
                        for c_idx in range(width):
                            norm_val = density_norm_resized[r_idx, c_idx]
                            rgba_color = matplotlib.cm.jet(norm_val) # Use jet colormap

                            kde_overlay_rgba_np[r_idx, c_idx, 0] = int(rgba_color[0] * 255)  # R
                            kde_overlay_rgba_np[r_idx, c_idx, 1] = int(rgba_color[1] * 255)  # G
                            kde_overlay_rgba_np[r_idx, c_idx, 2] = int(rgba_color[2] * 255)  # B
                            kde_overlay_rgba_np[r_idx, c_idx, 3] = int(norm_val * 200) # Alpha, max 200

                    kde_pil = Image.fromarray(kde_overlay_rgba_np, 'RGBA')
                    final_composite_pil = Image.alpha_composite(final_composite_pil, kde_pil)

                except Exception as e_kde:
                    print(f"KDE Heatmap generation failed: {e_kde}. Skipping KDE layer.") # Proper logging needed

            # Zone Layer
            zone_overlay_pil = Image.new('RGBA', final_composite_pil.size, (0,0,0,0))
            draw_context = ImageDraw.Draw(zone_overlay_pil)

            for zone in floorplan_data.get("zones", []):
                x, y, w, h = zone["x"], zone["y"], zone["width"], zone["height"]
                draw_context.rectangle([(x,y), (x+w, y+h)], outline=(0, 0, 255, 180), width=3) # Blue, semi-transparent
            
            final_composite_pil = Image.alpha_composite(final_composite_pil, zone_overlay_pil)
            
            # Save to bytes
            img_byte_arr = io.BytesIO()
            final_composite_pil.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            # Upload to Cloudinary
            folder = "meyraki/heatmaps"
            file_name = f"heatmap_{uuid.uuid4().hex}"
            
            heatmap_result = await cloudinary_service.upload_file(
                img_byte_arr.getvalue(),
                folder,
                file_name,
                "image"
            )
            
            return {
                "url": heatmap_result["secure_url"],
                "public_id": heatmap_result["public_id"],
                "width": width,
                "height": height
            }
            
        except Exception as e:
            raise Exception(f"Failed to generate heatmap: {str(e)}")
    
    @staticmethod
    async def generate_insights(project_id: str) -> Dict[str, Any]:
        """
        Generate insights for a project
        
        Args:
            project_id: Project ID
            
        Returns:
            Generated insights
        """
        try:
            # Get project data
            project = await project_service.get_project(project_id)
            
            if not project:
                raise Exception(f"Project with ID {project_id} not found")
            
            if not project.floorplan:
                raise Exception("Project has no floorplan uploaded")
            
            # Process floorplan
            floorplan_data = await AnalysisService.process_floorplan(project.floorplan.url)
            
            # Process usage data if available
            usage_data = {"has_coordinates": False, "hotspots": []}
            if project.usage_data:
                usage_data = await AnalysisService.process_usage_data(project.usage_data.url)
            
            # Generate heatmap
            heatmap_data = await AnalysisService.generate_heatmap(
                floorplan_data, 
                usage_data, 
                project.floorplan.url
            )
            
            # Generate recommendations based on objectives
            recommendations = await AnalysisService._generate_recommendations(
                project, 
                floorplan_data, 
                usage_data
            )
            
            # Generate PDF report
            report_data = await AnalysisService.generate_report(
                project,
                floorplan_data,
                usage_data,
                heatmap_data,
                recommendations
            )
            
            # Save insights to project
            insight_result = await project_service.add_insight_result(project_id, {
                "heatmap_url": heatmap_data["url"],
                "recommendations": recommendations,
                "report_url": report_data["url"]
            })
            
            return {
                "heatmap_url": heatmap_data["url"],
                "recommendations": recommendations,
                "report_url": report_data["url"]
            }
            
        except Exception as e:
            raise Exception(f"Failed to generate insights: {str(e)}")
    
    @staticmethod
    async def generate_report(project: Project, floorplan_data: Dict[str, Any], 
                         usage_data: Dict[str, Any], heatmap_data: Dict[str, Any],
                         recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a PDF report for the project
        
        Args:
            project: Project data
            floorplan_data: Processed floorplan data
            usage_data: Processed usage data
            heatmap_data: Generated heatmap data
            recommendations: Generated recommendations
            
        Returns:
            Report data including Cloudinary URL
        """
        try:
            # Create a PDF buffer
            buffer = io.BytesIO()
            
            # Set up the PDF document
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            
            # Custom styles
            title_style = ParagraphStyle(
                'Title',
                parent=styles['Title'],
                fontSize=24,
                spaceAfter=30
            )
            
            subtitle_style = ParagraphStyle(
                'Subtitle',
                parent=styles['Heading2'],
                fontSize=16,
                spaceAfter=15,
                spaceBefore=20
            )
            
            body_style = ParagraphStyle(
                'Body',
                parent=styles['Normal'],
                fontSize=12,
                spaceAfter=10
            )
            
            # Document content
            content = []
            
            # Title
            content.append(Paragraph(f"Meyraki Insight: {project.name}", title_style))
            
            # Project details
            content.append(Paragraph("Project Overview", subtitle_style))
            content.append(Paragraph(f"Space Type: {project.space_type.value}", body_style))
            
            objectives_text = ", ".join([obj.value for obj in project.objectives])
            content.append(Paragraph(f"Objectives: {objectives_text}", body_style))
            
            if project.description:
                content.append(Paragraph(f"Description: {project.description}", body_style))
            
            content.append(Spacer(1, 20))
            
            # Heatmap
            content.append(Paragraph("Space Analysis Heatmap", subtitle_style))
            response = urllib.request.urlopen(heatmap_data["url"])
            heatmap_img_data = response.read()
            img = RLImage(io.BytesIO(heatmap_img_data), width=450, height=300)
            content.append(img)
            content.append(Spacer(1, 10))
            
            # Efficiency metrics
            content.append(Paragraph("Space Efficiency", subtitle_style))
            
            efficiency = floorplan_data.get("efficiency", 0) * 100
            content.append(Paragraph(f"Space Utilization: {efficiency:.1f}%", body_style))
            
            # Recommendations
            content.append(Paragraph("Recommendations", subtitle_style))
            
            # Add each recommendation category
            for category, items in recommendations.items():
                content.append(Paragraph(f"{category.replace('_', ' ').title()}", ParagraphStyle(
                    'Category',
                    parent=styles['Heading3'],
                    fontSize=14,
                    spaceAfter=10,
                    spaceBefore=15
                )))
                
                # Add each recommendation item
                if isinstance(items, list):
                    for item in items:
                        content.append(Paragraph(f"• {item}", body_style))
                else:
                    content.append(Paragraph(f"{items}", body_style))
                    
            # Conclusion
            content.append(Paragraph("Conclusion", subtitle_style))
            content.append(Paragraph(
                "This report provides an analysis of your space based on the floor plan and usage data. "
                "We recommend implementing the suggested changes to optimize your space according to your objectives.",
                body_style
            ))
            
            # Build the PDF
            doc.build(content)
            
            # Get the PDF content
            pdf_content = buffer.getvalue()
            buffer.close()
            
            # Upload to Cloudinary
            folder = "meyraki/reports"
            file_name = f"report_{project.id}_{int(time.time())}"
            
            report_result = await cloudinary_service.upload_file(
                pdf_content,
                folder,
                file_name,
                "raw"
            )
            
            return {
                "url": report_result["secure_url"],
                "public_id": report_result["public_id"]
            }
            
        except Exception as e:
            raise Exception(f"Failed to generate report: {str(e)}")
    
    @staticmethod
    async def generate_moodboard(project_id: str) -> Dict[str, Any]:
        """
        Generate a moodboard for a project
        
        Args:
            project_id: Project ID
            
        Returns:
            Generated moodboard data
        """
        try:
            # Get project data
            project = await project_service.get_project(project_id)
            
            if not project:
                raise Exception(f"Project with ID {project_id} not found")
            
            # Based on the space type and objectives, create a simple moodboard
            # For MVP, we're using placeholder images based on space type
            
            # Determine style based on space type and objectives
            space_type = project.space_type
            
            # Map styles based on space type (simplified for MVP)
            style_map = {
                SpaceType.HOTEL: "Modern Luxury",
                SpaceType.OFFICE: "Corporate Minimalist",
                SpaceType.CAFE: "Cozy Industrial",
                SpaceType.RESTAURANT: "Elegant Dining",
                SpaceType.RETAIL: "Contemporary Retail",
                SpaceType.RESIDENTIAL: "Modern Living",
                SpaceType.OTHER: "Versatile Contemporary"
            }
            
            style = style_map.get(space_type, "Contemporary")
            
            # For objectives, we'll modify the style description
            if ObjectiveType.FLOW in project.objectives:
                style += " with optimized circulation"
            if ObjectiveType.AMBIANCE in project.objectives:
                style += " with atmospheric lighting"
            if ObjectiveType.REVENUE in project.objectives:
                style += " focused on revenue-generating zones"
            
            # For MVP, create a placeholder moodboard image
            # Download a sample image based on space type
            sample_image_url = f"https://source.unsplash.com/800x600/?{space_type.value},interior"
            
            response = urllib.request.urlopen(sample_image_url)
            image_data = response.read()
            
            # Upload to Cloudinary
            folder = "meyraki/moodboards"
            file_name = f"moodboard_{project.id}"
            
            moodboard_result = await cloudinary_service.upload_file(
                image_data,
                folder,
                file_name,
                "image"
            )
            
            # Save moodboard to project
            moodboard_data = {
                "moodboard_url": moodboard_result["secure_url"],
                "style_description": style
            }
            
            await project_service.add_moodboard_result(project_id, moodboard_data)
            
            return {
                "moodboard_url": moodboard_result["secure_url"],
                "style_description": style
            }
            
        except Exception as e:
            raise Exception(f"Failed to generate moodboard: {str(e)}")
    
    # Private helper methods
    @staticmethod
    async def _generate_recommendations(project: Project, floorplan_data: Dict[str, Any], 
                                   usage_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate recommendations based on project objectives and analysis data
        
        Args:
            project: Project data
            floorplan_data: Processed floorplan data
            usage_data: Processed usage data
            
        Returns:
            Recommendations dict
        """
        recommendations = {}
        
        # Base recommendations on objectives
        for objective in project.objectives:
            if objective == ObjectiveType.FLOW:
                recommendations["flow_recommendations"] = AnalysisService._get_flow_recommendations(
                    floorplan_data, usage_data, project.space_type
                )
            
            elif objective == ObjectiveType.AMBIANCE:
                recommendations["ambiance_recommendations"] = AnalysisService._get_ambiance_recommendations(
                    project.space_type
                )
            
            elif objective == ObjectiveType.ZONING:
                recommendations["zoning_recommendations"] = AnalysisService._get_zoning_recommendations(
                    floorplan_data, usage_data, project.space_type
                )
            
            elif objective == ObjectiveType.REVENUE:
                recommendations["revenue_recommendations"] = AnalysisService._get_revenue_recommendations(
                    usage_data, project.space_type
                )
            
            elif objective == ObjectiveType.EFFICIENCY:
                recommendations["efficiency_recommendations"] = AnalysisService._get_efficiency_recommendations(
                    floorplan_data, project.space_type
                )
        
        # Add general recommendations
        recommendations["general_recommendations"] = [
            "Ensure proper lighting throughout the space",
            "Consider accessibility needs in all areas",
            "Maintain clear emergency exits and pathways"
        ]
        
        return recommendations
    
    @staticmethod
    def _get_flow_recommendations(floorplan_data: Dict[str, Any], usage_data: Dict[str, Any], 
                             space_type: SpaceType) -> List[str]:
        """Generate flow recommendations"""
        recommendations = []
        zones = floorplan_data.get("zones", [])
        
        # Check hotspots for congestion
        if usage_data.get("has_coordinates") and usage_data.get("hotspots"):
            high_traffic_spots = [h for h in usage_data["hotspots"] if h["strength"] > 0.3]
            
            if high_traffic_spots:
                for hotspot in high_traffic_spots:
                    hx, hy = hotspot["x"], hotspot["y"]
                    found_in_zone = False
                    for zone in zones:
                        zx, zy, zw, zh = zone["x"], zone["y"], zone["width"], zone["height"]
                        if zx < hx < zx + zw and zy < hy < zy + zh:
                            recommendations.append(
                                f"High traffic area detected near Zone {zone['id']} (approx. center x={hx}, y={hy}). "
                                f"Consider widening pathways in or leading to this zone."
                            )
                            found_in_zone = True
                            break
                    if not found_in_zone:
                        recommendations.append(
                            f"High traffic area detected at coordinates (x={hx}, y={hy}). "
                            f"Consider widening pathways in this general area."
                        )
            elif not zones: # No high traffic but also no zones to reference
                 recommendations.append("Consider defining zones in your floorplan to better analyze foot traffic and flow.")

        elif zones: # No usage data, but zones exist
            zone_ids = [zone['id'] for zone in zones[:3]] # Example: Zone A, Zone B, Zone C
            if len(zone_ids) > 1 :
                 recommendations.append(
                    f"Ensure clear and wide enough pathways connect all identified major zones "
                    f"(e.g., {', '.join(zone_ids)})."
                )
            elif len(zone_ids) == 1:
                 recommendations.append(
                    f"Ensure clear and wide pathways within and leading to Zone {zone_ids[0]}."
                 )
        else: # No usage data and no zones
            recommendations.append("Upload usage data or define floorplan zones for more specific flow recommendations.")

        # Original space-type specific recommendations (can be kept or refined)
        if space_type == SpaceType.HOTEL:
            recommendations.extend([
                "Create a clear path from entrance to reception",
                "Ensure elevator lobbies have adequate waiting space",
                "Separate service circulation from guest flow"
            ])
        elif space_type == SpaceType.OFFICE:
            recommendations.extend([
                "Position frequently used resources (printers, supplies) in easily accessible locations",
                "Create buffer zones between focused work areas and high-traffic paths",
                "Consider one-way circulation for narrow corridors"
            ])
        elif space_type == SpaceType.CAFE:
            recommendations.extend([
                "Position ordering counter with clear queue space",
                "Separate takeaway customers from seated guests",
                "Ensure staff have efficient paths between service areas"
            ])
        else:
            recommendations.extend([
                "Create intuitive wayfinding through the space",
                "Allow for adequate circulation space in high-traffic areas",
                "Avoid creating dead-end paths that force backtracking"
            ])
            
        return recommendations
    
    @staticmethod
    def _get_ambiance_recommendations(floorplan_data: Dict[str, Any], space_type: SpaceType) -> List[str]:
        """Generate ambiance recommendations"""
        recommendations = []
        zones = floorplan_data.get("zones", [])
        
        # Generic ambiance recommendations
        recommendations.append("Layer lighting with ambient, task, and accent fixtures throughout the space.")
        recommendations.append("Use materials, colors, and textures that align with your brand identity and desired atmosphere for a {space_type.value}.")
        
        if zones:
            largest_zone = max(zones, key=lambda z: z['area'], default=None)
            if largest_zone:
                recommendations.append(
                    f"Pay particular attention to ambiance in key zones such as {largest_zone['id']} (largest identified zone). "
                    f"Also, consider high-traffic zones if identified in usage patterns."
                )
            else:
                recommendations.append(
                    "Apply these ambiance considerations systematically across all identified zones."
                )
        else:
            recommendations.append("Defining zones in your floorplan can help tailor ambiance strategies to specific areas.")

        # Original space-specific recommendations
        if space_type == SpaceType.HOTEL:
            recommendations.extend([
                "Create a memorable arrival experience with statement lighting and art",
                "Use soundproofing materials in guest rooms and corridors",
                "Consider scent strategy to create a signature hotel experience"
            ])
        elif space_type == SpaceType.OFFICE:
            recommendations.extend([
                "Incorporate biophilic elements like plants and natural materials",
                "Vary ceiling heights to create different zones and feelings",
                "Use color psychology to enhance productivity in different work areas"
            ])
        elif space_type == SpaceType.CAFE:
            recommendations.extend([
                "Create Instagram-worthy moments with feature walls or unique installations",
                "Consider acoustics - use soft materials to absorb sound in busy areas",
                "Adjust lighting throughout the day to match desired energy levels"
            ])
        elif space_type == SpaceType.RESTAURANT:
            recommendations.extend([
                "Design lighting to make food and guests look their best",
                "Create acoustic comfort with sound-absorbing materials",
                "Consider different 'moods' for different dining zones"
            ])
            
        return recommendations
    
    @staticmethod
    def _get_zoning_recommendations(floorplan_data: Dict[str, Any], usage_data: Dict[str, Any], 
                               space_type: SpaceType) -> List[str]: # project: Project could be added if objectives are needed
        """Generate zoning recommendations"""
        recommendations = []
        zones = floorplan_data.get("zones", [])
        total_occupied_area = floorplan_data.get("occupied_area", 1) # Avoid division by zero if no zones
        total_area = floorplan_data.get("total_area", 1)

        if not zones:
            recommendations.append("No zones identified. Define zones in your floorplan to get specific zoning advice.")
        else:
            zones_count = len(zones)
            if zones_count < 3:
                recommendations.append("Consider creating more distinct zones (at least 3) to better organize the space according to its functions.")
            elif zones_count > 8:
                recommendations.append(f"The space is divided into {zones_count} zones. If this feels too fragmented, consider simplifying or merging some to improve clarity and flow, depending on your operational needs for a {space_type.value} environment.")

            avg_zone_area = total_occupied_area / zones_count if zones_count > 0 else 0

            for zone in zones:
                zone_id = zone["id"]
                zone_area = zone["area"]
                # Check for very large zones
                if total_occupied_area > 0 and zone_area / total_occupied_area > 0.5: # Zone takes >50% of occupied area
                    recommendations.append(
                        f"Zone {zone_id} is very large, covering over 50% of the identified occupied space. "
                        f"Review if its current use effectively utilizes this entire area for a {space_type.value}, "
                        f"or if it could be subdivided to serve multiple functions or objectives."
                    )
                elif zones_count > 1 and avg_zone_area > 0 and zone_area > 3 * avg_zone_area: # Significantly larger than average
                     recommendations.append(
                        f"Zone {zone_id} is substantially larger (more than 3x the average zone size). "
                        f"Ensure its size is justified by its function within a {space_type.value} space, "
                        f"or consider if it can be optimized or partially allocated to other needs."
                    )

            # Check for many small zones (avg area < 5% of total floorplan area)
            if zones_count > 3 and avg_zone_area / total_area < 0.05 : # average zone area is < 5% of total floorplan area
                 recommendations.append(
                    f"There are {zones_count} zones identified, and the average zone size is relatively small "
                    f"(less than 5% of the total floorplan area). Ensure this level of division is necessary "
                    f"and effectively serves your objectives for a {space_type.value}. "
                    f"If not, consider combining some smaller zones for better flow or utility."
                )

        # Original space-specific zoning recommendations (can be kept or refined)
        if space_type == SpaceType.HOTEL:
            recommendations.extend([
                "Clearly separate public, semi-private, and private zones",
                "Create distinct arrival, social, and service zones",
                "Consider creating 'discovery' spaces that encourage exploration"
            ])
        elif space_type == SpaceType.OFFICE:
            recommendations.extend([
                "Establish zones for focus work, collaboration, and socialization",
                "Create buffer spaces between noisy and quiet areas",
                "Designate areas for informal meetings separate from workstations"
            ])
        elif space_type == SpaceType.CAFE:
            recommendations.extend([
                "Zone the space for different visit durations (quick coffee vs. longer stays)",
                "Create clear distinction between ordering, waiting, and consumption areas",
                "Consider solo vs. group seating zones"
            ])
            
        return recommendations
    
    @staticmethod
    def _get_revenue_recommendations(floorplan_data: Dict[str, Any], usage_data: Dict[str, Any], space_type: SpaceType) -> List[str]:
        """Generate revenue-focused recommendations"""
        recommendations = []
        zones = floorplan_data.get("zones", [])
        
        # Generic revenue recommendations
        recommendations.append("Position high-margin items/services in high-visibility locations. Analyze customer paths if usage data is available.")

        if zones:
            largest_zone = max(zones, key=lambda z: z['area'], default=None)
            # Attempt to find entrance zone if names are standardized, otherwise use largest or high traffic
            # This is a placeholder for more advanced zone semantic understanding
            entrance_zone_candidates = [z for z in zones if "entrance" in z.get("name", "").lower() or "lobby" in z.get("name", "").lower()]
            key_operational_zone = None
            if entrance_zone_candidates:
                key_operational_zone = entrance_zone_candidates[0]
            elif largest_zone:
                key_operational_zone = largest_zone

            if key_operational_zone:
                 recommendations.append(
                    f"Consider the strategic placement of revenue-generating elements relative to key areas like Zone {key_operational_zone['id']}. "
                    f"If usage data shows specific high-traffic zones, prioritize those."
                )
        else:
            recommendations.append("Defining functional zones (e.g., entrance, retail area, service points) can help optimize placement for revenue generation.")

        # Original space-specific revenue recommendations
        if space_type == SpaceType.HOTEL:
            recommendations.extend([
                "Create compelling upgrade paths through room design hierarchy",
                "Position retail/F&B outlets along main guest circulation routes",
                "Design accommodating spaces for revenue-generating events"
            ])
        elif space_type == SpaceType.CAFE or space_type == SpaceType.RESTAURANT:
            recommendations.extend([
                "Position high-margin items at eye level on menu boards",
                "Create seating that supports optimal table turn times for your concept",
                "Design layout to support upselling opportunities (visible dessert displays, etc.)"
            ])
        elif space_type == SpaceType.RETAIL:
            recommendations.extend([
                "Position high-margin products in prime visibility areas",
                "Create a layout that guides customers past promotional displays",
                "Design checkout areas to accommodate impulse purchase displays"
            ])
            
        return recommendations
    
    @staticmethod
    def _get_efficiency_recommendations(floorplan_data: Dict[str, Any], space_type: SpaceType) -> List[str]:
        """Generate efficiency recommendations"""
        recommendations = []
        
        efficiency = floorplan_data.get("efficiency", 0)
        zones = floorplan_data.get("zones", [])
        
        if efficiency < 0.5:
            base_message = "The overall space utilization (efficiency) is low ({:.1f}%). ".format(efficiency * 100)
            if zones:
                example_zone_ids = [z["id"] for z in zones[:2]] # Get up to 2 example zone IDs
                if example_zone_ids:
                    base_message += (f"Review the floor plan for large, undefined or underutilized areas. "
                                     f"These could potentially be assigned new functions, expanded from existing zones like {', '.join(example_zone_ids)}, "
                                     f"or developed into new zones to better serve your {space_type.value} objectives.")
                else: # Should not happen if zones exist, but as a fallback
                    base_message += ("Consider if all areas of the floorplan have a defined purpose. "
                                     "Assigning functions to unused spaces can improve overall efficiency.")
            else:
                base_message += ("Consider defining functional zones in your floorplan. This can help identify "
                                 "underutilized spaces and improve overall efficiency for your {space_type.value}.")
            recommendations.append(base_message)
        elif efficiency > 0.85: # Changed from 0.8 to 0.85 for very dense
            recommendations.append(
                "The space is very densely used ({:.1f}% efficiency). While high utilization can be good, ".format(efficiency * 100) +
                f"ensure this doesn't compromise comfort, flow, or safety, especially in a {space_type.value} setting. "
                "Verify if key areas or pathways need more breathing room."
            )
        
        # Original space-specific efficiency recommendations (can be kept or refined)
        if space_type == SpaceType.OFFICE:
            recommendations.extend([
                "Evaluate workstation-to-employee ratio based on daily occupancy patterns",
                "Consider multifunctional spaces that can adapt to different needs",
                "Position shared resources (printers, supplies) in centralized locations"
            ])
        elif space_type == SpaceType.RETAIL:
            recommendations.extend([
                "Analyze sales-per-square-foot across different zones",
                "Ensure adequate storage to maintain clean, shoppable sales floor",
                "Design flexible fixtures that can accommodate changing merchandise needs"
            ])
            
        return recommendations
            
            
analysis_service = AnalysisService() 