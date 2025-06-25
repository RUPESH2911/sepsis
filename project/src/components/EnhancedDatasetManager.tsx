import React, { useState, useRef } from 'react';
import { Upload, Trash2, FileText, AlertCircle, CheckCircle, Loader, Download, FileSpreadsheet } from 'lucide-react';
import { ExcelProcessor } from '../utils/excelProcessor';
import { DatasetInfo, DatasetRow } from '../types/dataset';

interface EnhancedDatasetManagerProps {
  datasets: DatasetInfo[];
  onDatasetUpload: (dataset: DatasetRow[], info: DatasetInfo) => void;
  onDatasetDelete: (datasetId: string) => void;
  selectedDatasetId?: string;
  onDatasetSelect: (datasetId: string) => void;
}

export const EnhancedDatasetManager: React.FC<EnhancedDatasetManagerProps> = ({
  datasets,
  onDatasetUpload,
  onDatasetDelete,
  selectedDatasetId,
  onDatasetSelect
}) => {
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setUploadProgress(0);
    setUploadError(null);

    try {
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => Math.min(prev + Math.random() * 15, 85));
      }, 200);

      const { data, info } = await ExcelProcessor.processExcelFile(file);
      
      clearInterval(progressInterval);
      setUploadProgress(90);

      // Validate dataset
      const validation = ExcelProcessor.validateDataset(data);
      if (!validation.isValid) {
        throw new Error(`Dataset validation failed: ${validation.errors.join(', ')}`);
      }

      setUploadProgress(100);
      onDatasetUpload(data, info);
      
      setTimeout(() => {
        setUploading(false);
        setUploadProgress(0);
      }, 1000);
      
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      setUploadError((error as Error).message);
      setUploading(false);
      setUploadProgress(0);
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getStatusIcon = (status: DatasetInfo['status']) => {
    switch (status) {
      case 'ready': return <CheckCircle className="w-4 h-4 text-success-600" />;
      case 'processing': return <Loader className="w-4 h-4 text-primary-600 animate-spin" />;
      case 'error': return <AlertCircle className="w-4 h-4 text-danger-600" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Upload Section */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Dataset Management</h3>
        
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-primary-400 transition-colors">
          <input
            ref={fileInputRef}
            type="file"
            accept=".xlsx,.xls,.csv"
            onChange={handleFileUpload}
            className="hidden"
            disabled={uploading}
          />
          
          {uploading ? (
            <div className="space-y-4">
              <Loader className="w-12 h-12 text-primary-600 animate-spin mx-auto" />
              <div>
                <div className="text-lg font-medium text-gray-900">Processing Dataset...</div>
                <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                  <div
                    className="bg-primary-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
                <div className="text-sm text-gray-600 mt-1">{uploadProgress.toFixed(0)}% complete</div>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <FileSpreadsheet className="w-12 h-12 text-gray-400 mx-auto" />
              <div>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="text-lg font-medium text-primary-600 hover:text-primary-700"
                >
                  Upload Excel/CSV Dataset
                </button>
                <p className="text-gray-600 mt-1">
                  Click to select an Excel (.xlsx, .xls) or CSV file with patient data
                </p>
              </div>
              <div className="text-sm text-gray-500">
                Required columns: Patient_ID, Hour, vital signs, lab values, etc.
              </div>
            </div>
          )}
        </div>

        {uploadError && (
          <div className="mt-4 p-4 bg-danger-50 border border-danger-200 rounded-lg">
            <div className="flex items-center space-x-2">
              <AlertCircle className="w-5 h-5 text-danger-600" />
              <span className="font-medium text-danger-800">Upload Error</span>
            </div>
            <p className="text-sm text-danger-700 mt-1">{uploadError}</p>
          </div>
        )}
      </div>

      {/* Datasets List */}
      {datasets.length > 0 && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Uploaded Datasets</h3>
          
          <div className="space-y-3">
            {datasets.map((dataset) => (
              <div
                key={dataset.id}
                className={`p-4 border rounded-lg cursor-pointer transition-all ${
                  selectedDatasetId === dataset.id
                    ? 'border-primary-500 bg-primary-50'
                    : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                }`}
                onClick={() => onDatasetSelect(dataset.id)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <FileText className="w-5 h-5 text-gray-600" />
                    <div>
                      <div className="font-medium text-gray-900">{dataset.name}</div>
                      <div className="text-sm text-gray-600">
                        {dataset.rowCount.toLocaleString()} rows × {dataset.columnCount} columns
                      </div>
                      <div className="text-xs text-gray-500">
                        {dataset.patientIds.length} patients • Uploaded {dataset.uploadDate.toLocaleDateString()} • {formatFileSize(dataset.size)}
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-3">
                    {getStatusIcon(dataset.status)}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onDatasetDelete(dataset.id);
                      }}
                      className="p-2 text-gray-400 hover:text-danger-600 hover:bg-danger-50 rounded-lg transition-colors"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
                
                {dataset.targetColumn && (
                  <div className="mt-2 text-sm">
                    <span className="text-gray-600">Target column: </span>
                    <span className="font-medium text-primary-600">{dataset.targetColumn}</span>
                  </div>
                )}
                
                {selectedDatasetId === dataset.id && (
                  <div className="mt-3 pt-3 border-t border-gray-200">
                    <div className="text-sm text-gray-600 mb-2">
                      <strong>Sample Patient IDs:</strong> {dataset.patientIds.slice(0, 5).join(', ')}
                      {dataset.patientIds.length > 5 && ` +${dataset.patientIds.length - 5} more`}
                    </div>
                    <div className="text-sm text-gray-600">
                      <strong>Features:</strong> {dataset.features.slice(0, 8).join(', ')}
                      {dataset.features.length > 8 && ` +${dataset.features.length - 8} more`}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Sample Dataset Download */}
      <div className="bg-primary-50 rounded-lg border border-primary-200 p-6">
        <div className="flex items-center space-x-3">
          <Download className="w-5 h-5 text-primary-600" />
          <div>
            <h4 className="font-medium text-primary-900">Need a sample dataset?</h4>
            <p className="text-sm text-primary-700 mt-1">
              Download our sample sepsis dataset in Excel format to test the system
            </p>
          </div>
        </div>
        <button className="mt-3 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors">
          Download Sample Excel Dataset
        </button>
      </div>
    </div>
  );
};