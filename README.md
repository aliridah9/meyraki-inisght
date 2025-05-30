# Meyraki Insight

Meyraki Insight is an AI-powered spatial intelligence platform for hospitality and commercial interiors. The platform analyzes floorplans and usage data to provide recommendations for optimizing spaces based on specified objectives.

## Features

- Upload floorplans and usage data
- Define objectives for space optimization
- Generate AI-powered insights and recommendations
- View interactive heatmaps of space usage
- Generate PDF reports with actionable recommendations
- Create AI-generated moodboards for design inspiration

## Tech Stack

### Backend
- **FastAPI** (Python): API framework
- **MongoDB**: Database (using Motor for async access)
- **Cloudinary**: File storage for floorplans, heatmaps, and reports
- **OpenCV & scikit-learn**: Computer vision and machine learning

### Frontend
- **React**: UI framework
- **TypeScript**: Type-safe JavaScript
- **Material-UI**: Component library
- **React Query**: Data fetching and state management

### Deployment
- **Render**: Cloud hosting for frontend and backend

## Project Structure

```
.
├── backend/                  # FastAPI backend
│   ├── app/                  # Application code
│   │   ├── api/              # API endpoints
│   │   ├── core/             # Core functionality
│   │   ├── db/               # Database connections
│   │   ├── models/           # Pydantic models
│   │   ├── services/         # Business logic
│   │   └── utils/            # Utilities
│   ├── Dockerfile            # Docker configuration
│   └── requirements.txt      # Python dependencies
│
└── frontend/                 # React frontend
    ├── public/               # Static files
    └── src/                  # Source code
        ├── components/       # Reusable UI components
        ├── pages/            # Page components
        ├── services/         # API services
        └── utils/            # Utility functions
```

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 16+
- MongoDB instance (local or Atlas)
- Cloudinary account

### Backend Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/meyraki-insight.git
   cd meyraki-insight
   ```

2. Set up a virtual environment:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the backend directory (copy from env.template):
   ```
   # MongoDB Configuration
   MONGODB_CONNECTION_STRING=your_mongodb_connection_string (e.g., mongodb://localhost:27017 or Atlas URI)
   MONGODB_DATABASE_NAME=your_database_name (e.g., meyraki_db)

   # JWT Configuration
   JWT_SECRET_KEY=your_new_strong_secret_key_for_jwt

   # Cloudinary Configuration
   CLOUDINARY_CLOUD_NAME=your_cloudinary_cloud_name
   CLOUDINARY_API_KEY=your_cloudinary_api_key
   CLOUDINARY_API_SECRET=your_cloudinary_api_secret

   # App Configuration
   APP_NAME=Meyraki Insight
   BACKEND_CORS_ORIGINS=["http://localhost:3000","http://localhost:8000"]
   SECRET_KEY=your_secret_key_for_token_generation
   ```

5. Run the backend:
   ```bash
   uvicorn app.main:app --reload
   ```
   The API will be available at http://localhost:8000 with Swagger documentation at http://localhost:8000/docs

### Frontend Setup

1. Install dependencies:
   ```bash
   cd frontend
   npm install
   ```

2. Create a `.env` file in the frontend directory:
   ```
   REACT_APP_API_URL=http://localhost:8000/api/v1
   ```
   (Note: Frontend integration with the new backend authentication and data source will require separate updates to the frontend code.)

3. Run the frontend:
   ```bash
   npm start
   ```
   The app will be available at http://localhost:3000

## Database Setup

### MongoDB Collections

The application now uses MongoDB. The following collections are used:

*   **users**: Stores user information (authentication and user details).
*   **projects**: Stores project details.
*   **floorplans**: Stores metadata for uploaded floorplan files.
*   **usage_data_files**: Stores metadata for uploaded usage data files.
*   **insight_results**: Stores results of insight generation.
*   **moodboard_results**: Stores generated moodboard details.

Data validation and schema enforcement are primarily handled by Pydantic models within the FastAPI application. Refer to files in `backend/app/models/` for schema details.

## API Documentation

The API provides the following endpoints:

### Projects

- `POST /api/v1/projects`: Create a new project
- `GET /api/v1/projects`: Get all projects for the authenticated user
- `GET /api/v1/projects/{project_id}`: Get a specific project
- `PATCH /api/v1/projects/{project_id}`: Update a project
- `DELETE /api/v1/projects/{project_id}`: Delete a project
- `POST /api/v1/projects/{project_id}/upload-floorplan`: Upload a floorplan file
- `POST /api/v1/projects/{project_id}/upload-usage-data`: Upload usage data CSV
- `POST /api/v1/projects/{project_id}/set-objectives`: Set project objectives

### Insights

- `POST /api/v1/insights/{project_id}/generate`: Generate insights
- `GET /api/v1/insights/{project_id}`: Get existing insights for a project
- `POST /api/v1/insights/{project_id}/generate-moodboard`: Generate a moodboard
- `GET /api/v1/insights/{project_id}/download-report`: Get report download URL

## Deployment

### Backend Deployment on Render

1. Add your project to GitHub
2. Create a new Web Service on Render
3. Connect to your GitHub repository
4. Configure the service:
   - Build Command: `pip install -r backend/requirements.txt`
   - Start Command: `cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables as defined in `.env`

### Frontend Deployment on Render

1. Create a new Static Site on Render
2. Connect to your GitHub repository
3. Configure the service:
   - Build Command: `cd frontend && npm install && npm run build`
   - Publish Directory: `frontend/build`
4. Add environment variables as defined in `.env`

## License

This project is licensed under the MIT License - see the LICENSE file for details. 