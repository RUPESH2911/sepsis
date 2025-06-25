import React, { useState, useRef } from 'react';
import { Upload, Trash2, FileText, AlertCircle, CheckCircle, Loader, Download } from 'lucide-react';
import Papa from 'papaparse';
import { DatasetInfo, DatasetRow } from '../types/dataset';

interface DatasetManagerProps {
  datasets: DatasetInfo[];
  onDatasetUpload: (dataset: DatasetRow[], info: DatasetInfo) => void;
  onDatasetDelete: (datasetId: string) => void;
  selectedDatasetId?: string;
  onDatasetSelect: (datasetId: string) => void;
}

export const DatasetManager: React.FC<DatasetManagerProps> = ({
  datasets,
  onDatasetUpload,
  onDatasetDelete,
  selectedDatasetId,
  onDatasetSelect
}) => {
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setUploadProgress(0);

    try {
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => Math.min(prev + Math.random() * 20, 90));
      }, 200);

      Papa.parse(file, {
        header: true,
        skipEmptyLines: true,
        complete: (results) => {
          clearInterval(progressInterval);
          setUploadProgress(100);

          const data = results.data as DatasetRow[];
          const features = Object.keys(data[0] || {});
          
          const datasetInfo: DatasetInfo = {
            id: Date.now().toString(),
            name: file.name.replace('.csv', ''),
            uploadDate: new Date(),
            size: file.size,
            rowCount: data.length,
            columnCount: features.length,
            status: 'ready',
            features,
            targetColumn: features.includes('SepsisLabel') ? 'SepsisLabel' : undefined
          };

          onDatasetUpload(data, datasetInfo);
          setUploading(false);
          setUploadProgress(0);
          
          if (fileInputRef.current) {
            fileInputRef.current.value = '';
          }
        },
        error: (error) => {
          clearInterval(progressInterval);
          console.error('Error parsing CSV:', error);
          setUploading(false);
          setUploadProgress(0);
        }
      });
    } catch (error) {
      console.error('Error uploading file:', error);
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
            accept=".csv"
            onChange={handleFileUpload}
            className="hidden"
            disabled={uploading}
          />
          
          {uploading ? (
            <div className="space-y-4">
              <Loader className="w-12 h-12 text-primary-600 animate-spin mx-auto" />
              <div>
                <div className="text-lg font-medium text-gray-900">Uploading Dataset...</div>
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
              <Upload className="w-12 h-12 text-gray-400 mx-auto" />
              <div>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="text-lg font-medium text-primary-600 hover:text-primary-700"
                >
                  Upload CSV Dataset
                </button>
                <p className="text-gray-600 mt-1">
                  Click to select a CSV file with patient data
                </p>
              </div>
              <div className="text-sm text-gray-500">
                Expected columns: Patient_ID, Hour, vitals, labs, SepsisLabel, etc.
              </div>
            </div>
          )}
        </div>
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
                        Uploaded {dataset.uploadDate.toLocaleDateString()} • {formatFileSize(dataset.size)}
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
                    <div className="text-sm text-gray-600">
                      <strong>Features:</strong> {dataset.features.slice(0, 5).join(', ')}
                      {dataset.features.length > 5 && ` +${dataset.features.length - 5} more`}
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
              Download our sample sepsis dataset to test the system
            </p>
          </div>
        </div>
        <button className="mt-3 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors">
          Download Sample Dataset
        </button>
      </div>
    </div>
  );
};