import { BrowserRouter, Route, Routes } from 'react-router-dom';
import { AuthProvider } from './contexts/AuthContext';
import { TenantProvider } from './contexts/TenantContext';
import ProtectedRoute from './components/ProtectedRoute';
import AppLayout from './components/AppLayout';
import LoginPage from './pages/LoginPage';
import DashboardPage from './pages/DashboardPage';
import PointPickerPage from './pages/PointPickerPage';
import LinePickerPage from './pages/LinePickerPage';
import CameraAnnotatorPage from './pages/CameraAnnotatorPage';
import CameraLineAnnotatorPage from './pages/CameraLineAnnotatorPage';
import HomographyPrecisionPage from './pages/HomographyPrecisionPage';
import CameraDiagnosticPage from './pages/CameraDiagnosticPage';
import CameraSurveyPage from './pages/CameraSurveyPage';
import CameraEvaluationPage from './pages/CameraEvaluationPage';
import LensCalibrationPage from './pages/LensCalibrationPage';
import DistortionValidatorPage from './pages/DistortionValidatorPage';
import CleanPlateGalleryPage from './pages/CleanPlateGalleryPage';

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route element={<ProtectedRoute />}>
            <Route
              element={
                <TenantProvider>
                  <AppLayout />
                </TenantProvider>
              }
            >
              <Route index element={<DashboardPage />} />
              <Route path="point-picker" element={<PointPickerPage />} />
              <Route path="line-picker" element={<LinePickerPage />} />
              <Route
                path="camera-annotator"
                element={<CameraAnnotatorPage />}
              />
              <Route
                path="camera-line-annotator"
                element={<CameraLineAnnotatorPage />}
              />
              <Route
                path="homography-precision"
                element={<HomographyPrecisionPage />}
              />
              <Route
                path="camera-diagnostic"
                element={<CameraDiagnosticPage />}
              />
              <Route path="camera-survey" element={<CameraSurveyPage />} />
              <Route
                path="camera-evaluation"
                element={<CameraEvaluationPage />}
              />
              <Route
                path="lens-calibration"
                element={<LensCalibrationPage />}
              />
              <Route
                path="distortion-validator"
                element={<DistortionValidatorPage />}
              />
              <Route
                path="clean-plate-gallery"
                element={<CleanPlateGalleryPage />}
              />
            </Route>
          </Route>
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}
