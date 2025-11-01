# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Final Year Project (FYP) for a **Bitcoin Anti-Money Laundering (AML) Detection System** using Graph Neural Networks (GNNs). The project consists of two main components:

1. **Machine Learning Pipeline** (Jupyter Notebook): Data exploration, model training (GCN, GAT, Graph Transformer), and model evaluation
2. **Flutter Mobile Application** (`aml_app/`): Cross-platform app for analysts to review flagged Bitcoin transactions

## Dataset Structure

Located in `dataset/` directory:
- `txs_features.csv` - 203,769 Bitcoin transactions with 182 features (694MB)
- `txs_classes.csv` - Transaction labels (1=illicit, 2=licit, 3=unknown)
- `txs_edgelist.csv` - Transaction-to-transaction edges (234,355 edges)
- `wallets_features.csv` - Wallet-level features (606MB)
- `wallets_classes.csv` - Wallet labels
- `AddrAddr_edgelist.csv` - Address-to-address network (200MB)
- `AddrTx_edgelist.csv` / `TxAddr_edgelist.csv` - Address-transaction mappings

**Dataset Characteristics:**
- 49 timesteps (temporal data)
- Temporal split: timesteps 1-34 for training, 35-49 for testing
- Class distribution: 157,205 unknown, 42,019 licit, 4,545 illicit
- Highly imbalanced dataset requiring class weights

## Machine Learning Models

### Model Architecture (Main.ipynb)

Three Graph Neural Network models trained using PyTorch Geometric:

1. **GCN (Graph Convolutional Network)**
   - 2-layer GCN with 64 hidden dimensions
   - Test accuracy: 74.38%, F1: 0.2906

2. **GAT (Graph Attention Network)**
   - 2-layer GAT with 4 attention heads
   - Test accuracy: 79.15%, F1: 0.3248

3. **Graph Transformer** ⭐ (Best Model)
   - 2-layer Transformer with 4 heads
   - Test accuracy: 81.66%, F1: 0.3654
   - Saved to `models/graph_transformer.pt`

### Training Configuration

```python
# Data preprocessing
- MinMaxScaler fitted only on training data (timesteps 1-34)
- NaN values replaced with column median
- Binary classification: Illicit (1) vs Licit (0)

# Training hyperparameters
- Learning rate: 0.005
- Weight decay: 5e-4
- Dropout: 0.5
- Max epochs: 200
- Gradient clipping: max_norm=1.0
- Class weights applied to handle imbalance
```

### Running the Notebook

The notebook (`Main.ipynb`) contains all code for:
- Data exploration and visualization
- Feature preprocessing and temporal splitting
- Model training and evaluation
- Confusion matrix analysis
- Model comparison and performance metrics

To re-run experiments or modify models, work directly in `Main.ipynb`. The notebook outputs visualizations and saves the best model to `models/`.

## Flutter Application (`aml_app/`)

### Architecture

**State Management:** Provider pattern (not yet fully implemented)

**Backend:** Supabase (PostgreSQL database with real-time subscriptions)

**Key Directories:**
- `lib/screens/` - UI screens (analyst dashboard, guest view, transaction details)
  - `analyst/` - Analyst interface with review queue and transaction details
  - `guest/` - Read-only public dashboard
- `lib/services/` - Backend integration
  - `supabase_service.dart` - Database queries and real-time streams
  - `api_service.dart` - ML model inference API (currently empty/stub)
- `lib/models/` - Data models (Transaction, AnalystReview)
- `lib/config/` - Configuration (Supabase credentials)

### Development Commands

```bash
# Navigate to Flutter app
cd aml_app

# Install dependencies
flutter pub get

# Run app (choose platform)
flutter run                    # Interactive device selection
flutter run -d chrome          # Web browser
flutter run -d windows         # Windows desktop
flutter run -d <device-id>     # Specific device

# Build for production
flutter build apk              # Android
flutter build ios              # iOS (requires macOS)
flutter build web              # Web
flutter build windows          # Windows desktop

# Run tests
flutter test

# Analyze code
flutter analyze
```

### Database Schema (Supabase)

**transactions table:**
- `transaction_id` (PK, text)
- `timestamp` (timestamptz)
- `amount` (numeric, nullable)
- `features` (jsonb) - 182 transaction features
- `prediction_score` (numeric) - Model confidence score
- `predicted_label` (text) - 'illicit' or 'licit'
- `model_version` (text)
- `status` (text) - 'pending_review', 'reviewed', 'cleared'
- `created_at`, `updated_at`

**analyst_reviews table:**
- `review_id` (PK, uuid)
- `transaction_id` (FK)
- `analyst_id` (text)
- `manual_label` (text) - Analyst's verdict
- `notes` (text)
- `confidence` (text) - 'low', 'medium', 'high'
- `reviewed_at` (timestamptz)

**Key RPC function:** `get_dashboard_stats()` - Returns aggregated statistics

### Real-time Features

The app uses Supabase real-time subscriptions:
```dart
SupabaseService.watchFlaggedTransactions()
```
This provides live updates when new transactions are flagged by the ML model.

## Important Notes

### Model Integration
The Flutter app currently has a stub `api_service.dart`. The Graph Transformer model (`models/graph_transformer.pt`) needs to be deployed as a REST API (e.g., FastAPI, Flask) for the mobile app to consume. The model expects:
- Input: 182 features (normalized with saved scaler parameters)
- Output: Binary classification (illicit/licit) + confidence score

### Configuration
Supabase credentials are stored in `aml_app/lib/config/supabase_config.dart`. This file should be gitignored in production and use environment variables.

### Dataset Size
Dataset files are large (total ~2.2GB). Ensure adequate disk space and consider using `.gitignore` to exclude data files from version control.

### Dependencies
- **Python**: PyTorch, PyTorch Geometric, pandas, numpy, matplotlib, seaborn, plotly, scikit-learn
- **Flutter**: SDK 3.9.2+, packages listed in `aml_app/pubspec.yaml`

## Git Workflow

Current branch: `main`

The repository uses GitHub Actions for automated PR labeling (`.github/workflows/label.yml`).

When committing changes:
- ML experiments: Update `Main.ipynb` and commit saved models to `models/`
- Flutter app: Test on multiple platforms before committing
- Never commit Supabase credentials or API keys
