# HyperSyncDT: Advanced Digital Twin System

HyperSyncDT is a state-of-the-art Digital Twin system designed for industrial equipment monitoring and predictive maintenance. It combines advanced machine learning techniques with real-time sensor data to provide accurate predictions and anomaly detection.

## Features

- Real-time sensor data monitoring and visualization
- Advanced predictive maintenance using hybrid ML models
- Anomaly detection with multi-model ensemble approach
- Digital Twin simulation capabilities
- Interactive web interface with modern design
- RESTful API for system integration
- PostgreSQL database for robust data storage
- MLflow integration for experiment tracking

## System Architecture

The system consists of two main components:

### Backend (FastAPI)
- RESTful API endpoints for data access and control
- Machine learning models for prediction and anomaly detection
- PostgreSQL database integration
- Real-time data processing pipeline

### Frontend (Streamlit)
- Interactive dashboards for data visualization
- Real-time monitoring interface
- Configuration and control panels
- System status and health monitoring

## Prerequisites

- Python 3.11 or later
- PostgreSQL 13 or later
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hypersyncdt.git
cd hypersyncdt
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

4. Set up the environment variables:
```bash
cp backend/.env.example backend/.env
# Edit .env with your configuration
```

5. Initialize the database:
```bash
cd backend
python -c "from database.models import create_tables; create_tables()"
```

## Running the Application

1. Start the backend server:
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

2. Start the frontend application:
```bash
cd frontend
streamlit run app.py
```

The application will be available at:
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## API Documentation

The API provides the following main endpoints:

- `/machines`: Machine management and monitoring
- `/digital-twins`: Digital Twin operations and simulation
- `/predict/maintenance`: Predictive maintenance endpoints
- `/detect/anomalies`: Anomaly detection services

For detailed API documentation, visit http://localhost:8000/docs when the server is running.

## Development

### Project Structure
```
hypersyncdt/
├── backend/
│   ├── app/
│   │   └── ml/
│   │       ├── predictive_maintenance.py
│   │       └── anomaly_detector.py
│   ├── database/
│   │   ├── config.py
│   │   └── models.py
│   ├── main.py
│   └── requirements.txt
├── frontend/
│   ├── app.py
│   └── requirements.txt
└── README.md
```

### Code Style

The project follows PEP 8 guidelines. We use:
- `black` for code formatting
- `isort` for import sorting
- `flake8` for linting

To format the code:
```bash
black .
isort .
flake8
```

## Testing

Run the test suite:
```bash
pytest
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- FastAPI for the powerful API framework
- Streamlit for the interactive frontend
- PyTorch for machine learning capabilities
- SQLAlchemy for database operations

## Support

For support, please open an issue in the GitHub repository or contact the maintainers. 