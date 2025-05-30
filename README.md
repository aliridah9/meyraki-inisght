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
- **Supabase** (PostgreSQL): Database and authentication
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
- Supabase account
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
   # Supabase Configuration
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   SUPABASE_JWT_SECRET=your_supabase_jwt_secret

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
   REACT_APP_SUPABASE_URL=your_supabase_url
   REACT_APP_SUPABASE_ANON_KEY=your_supabase_anon_key
   ```

3. Run the frontend:
   ```bash
   npm start
   ```
   The app will be available at http://localhost:3000

## Database Setup

### Supabase Tables Structure

Create the following tables in your Supabase project:

1. **projects**
   ```sql
   create table projects (
     id uuid primary key,
     user_id uuid not null,
     name text not null,
     description text,
     space_type text not null,
     objectives text[] not null,
     custom_objectives text[],
     additional_notes text,
     status text not null,
     created_at timestamptz not null,
     updated_at timestamptz
   );
   ```

2. **floorplans**
   ```sql
   create table floorplans (
     project_id uuid primary key references projects(id) on delete cascade,
     file_id text not null,
     filename text not null,
     url text not null,
     file_type text not null,
     created_at timestamptz not null,
     size int not null,
     width int,
     height int
   );
   ```

3. **usage_data**
   ```sql
   create table usage_data (
     project_id uuid primary key references projects(id) on delete cascade,
     file_id text not null,
     filename text not null,
     url text not null,
     created_at timestamptz not null,
     size int not null,
     row_count int,
     column_count int
   );
   ```

4. **insight_results**
   ```sql
   create table insight_results (
     project_id uuid primary key references projects(id) on delete cascade,
     heatmap_url text,
     recommendations jsonb not null,
     report_url text,
     created_at timestamptz not null,
     updated_at timestamptz
   );
   ```

5. **moodboard_results**
   ```sql
   create table moodboard_results (
     project_id uuid primary key references projects(id) on delete cascade,
     moodboard_url text not null,
     style_description text not null,
     created_at timestamptz not null
   );
   ```

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