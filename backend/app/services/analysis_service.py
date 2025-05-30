import io
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import urllib.request
from typing import Dict, Any, List, Optional, Tuple
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
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
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
            
            # Calculate total floor area
            height, width = img.shape[:2]
            total_area = width * height
            
            # Calculate space efficiency
            occupied_area = sum(zone["area"] for zone in zones)
            efficiency = occupied_area / total_area if total_area > 0 else 0
            
            return {
                "width": width,
                "height": height,
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
            
            return {
                "row_count": row_count,
                "column_count": column_count,
                "has_coordinates": has_coordinates,
                "hotspots": hotspots,
                "columns": list(df.columns)
            }
            
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
            img = Image.open(io.BytesIO(image_data))
            width, height = img.size
            
            # Create a transparent overlay for the heatmap
            heatmap = np.zeros((height, width, 4), dtype=np.uint8)
            
            # Draw hotspots on the heatmap
            if usage_data.get("has_coordinates"):
                for hotspot in usage_data.get("hotspots", []):
                    # Get hotspot properties
                    x, y = hotspot["x"], hotspot["y"]
                    strength = hotspot["strength"]
                    
                    # Ensure coordinates are within image bounds
                    if 0 <= x < width and 0 <= y < height:
                        # Draw a gradient circle for each hotspot
                        # Stronger hotspots have larger radius
                        radius = int(50 * strength) + 20
                        
                        # Red color with alpha based on strength
                        color = (255, 0, 0, int(200 * strength))
                        
                        # Draw circle on the heatmap
                        cv2.circle(heatmap, (x, y), radius, color, -1)
            
            # Add zones from floorplan analysis
            for zone in floorplan_data.get("zones", []):
                x, y, w, h = zone["x"], zone["y"], zone["width"], zone["height"]
                
                # Draw semi-transparent blue rectangle for each zone
                cv2.rectangle(heatmap, (x, y), (x + w, y + h), (0, 0, 255, 60), 2)
            
            # Convert NumPy array to PIL Image
            heatmap_img = Image.fromarray(heatmap, 'RGBA')
            
            # Overlay heatmap on floorplan
            result = Image.alpha_composite(img.convert('RGBA'), heatmap_img)
            
            # Save to bytes
            img_byte_arr = io.BytesIO()
            result.save(img_byte_arr, format='PNG')
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
        
        # Check hotspots for congestion
        if usage_data.get("has_coordinates") and usage_data.get("hotspots"):
            # Find hotspots with high activity
            high_traffic_spots = [h for h in usage_data["hotspots"] if h["strength"] > 0.3]
            
            if high_traffic_spots:
                recommendations.append(f"High traffic areas detected at several points. Consider widening pathways in these areas.")
        
        # Check for zone-specific recommendations
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
    def _get_ambiance_recommendations(space_type: SpaceType) -> List[str]:
        """Generate ambiance recommendations"""
        recommendations = []
        
        # Generic ambiance recommendations
        recommendations.append("Layer lighting with ambient, task, and accent fixtures")
        recommendations.append("Use materials and textures that align with your brand identity")
        
        # Space-specific recommendations
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
                               space_type: SpaceType) -> List[str]:
        """Generate zoning recommendations"""
        recommendations = []
        
        zones_count = len(floorplan_data.get("zones", []))
        
        if zones_count < 3:
            recommendations.append("Consider creating more distinct zones to better organize the space")
        elif zones_count > 8:
            recommendations.append("The space may have too many separate zones. Consider simplifying to improve clarity")
        
        # Space-specific zoning recommendations
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
    def _get_revenue_recommendations(usage_data: Dict[str, Any], space_type: SpaceType) -> List[str]:
        """Generate revenue-focused recommendations"""
        recommendations = []
        
        # Generic revenue recommendations
        recommendations.append("Position high-margin items/services in high-visibility locations")
        
        # Space-specific revenue recommendations
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
        
        if efficiency < 0.5:
            recommendations.append("The space utilization is low. Consider consolidating zones to reduce wasted space")
        elif efficiency > 0.8:
            recommendations.append("The space is very densely used. Consider creating more breathing room in key areas")
        
        # Space-specific efficiency recommendations
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