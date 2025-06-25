import React, { useState } from 'react';
import Papa from 'papaparse';
import * as XLSX from 'xlsx';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  CartesianGrid
} from 'recharts';

// Placeholder: Replace this with your real model prediction function or API call
function predictSepsis(data: Array<Record<string, any>>): number[] {
  // Example: Random predictions (replace with real model logic)
  return data.map(() => Math.random());
}

interface CombinedDatum {
  index: number;
  actual: number;
  predicted: number;
}

function parseFile(file: File, callback: (data: Array<Record<string, any>>) => void, onError: (msg: string) => void) {
  const name = file.name.toLowerCase();
  if (name.endsWith('.csv')) {
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: (results) => callback(results.data as Array<Record<string, any>>),
      error: (err) => onError('Failed to parse CSV: ' + err.message)
    });
  } else if (name.endsWith('.xlsx') || name.endsWith('.xls')) {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const data = new Uint8Array(e.target?.result as ArrayBuffer);
        const workbook = XLSX.read(data, { type: 'array' });
        const sheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[sheetName];
        const jsonData = XLSX.utils.sheet_to_json(worksheet, { defval: '', raw: false });
        callback(jsonData as Array<Record<string, any>>);
      } catch (err: any) {
        onError('Failed to parse Excel: ' + err.message);
      }
    };
    reader.onerror = () => onError('Failed to read Excel file');
    reader.readAsArrayBuffer(file);
  } else {
    onError('Unsupported file type. Please upload a CSV or Excel file.');
  }
}

const SepsisModelPerformance: React.FC = () => {
  const [unknownData, setUnknownData] = useState<Array<Record<string, any>>>([]);
  const [actualData, setActualData] = useState<Array<Record<string, any>>>([]);
  const [combined, setCombined] = useState<CombinedDatum[]>([]);
  const [error, setError] = useState<string | null>(null);

  // Handle file upload and parse CSV/Excel
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>, type: 'unknown' | 'actual') => {
    const file = e.target.files?.[0];
    if (!file) return;
    parseFile(
      file,
      (data) => {
        if (type === 'unknown') {
          setUnknownData(data);
        } else {
          setActualData(data);
        }
      },
      (msg) => setError(msg)
    );
  };

  // Combine and predict when both datasets are loaded
  React.useEffect(() => {
    setError(null);
    if (unknownData.length === 0 || actualData.length === 0) {
      setCombined([]);
      return;
    }
    // Ensure columns line up (basic check)
    const unknownKeys = Object.keys(unknownData[0] || {});
    const actualKeys = Object.keys(actualData[0] || {});
    if (!unknownKeys.every(k => actualKeys.includes(k) || k === 'SepsisLabel')) {
      setError('Column mismatch between unknown and actual datasets.');
      setCombined([]);
      return;
    }
    // Predict (replace with real model call)
    const predicted = predictSepsis(unknownData);
    // Extract actual labels (ensure numeric)
    const actual = actualData.map(row => Number(row.SepsisLabel));
    // Build combined array
    const minLen = Math.min(predicted.length, actual.length);
    const combinedArr: CombinedDatum[] = [];
    for (let i = 0; i < minLen; i++) {
      combinedArr.push({
        index: i + 1,
        actual: actual[i],
        predicted: predicted[i]
      });
    }
    setCombined(combinedArr);
  }, [unknownData, actualData]);

  // Detect if X-axis labels overlap (simple heuristic)
  const rotateLabels = combined.length > 20;

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold mb-4">Sepsis Model Performance: Predicted vs Actual</h2>
      <div className="flex flex-col md:flex-row gap-6 mb-4">
        <div>
          <label className="block font-medium mb-2">Upload Unknown Data (features only):</label>
          <input type="file" accept=".csv,.xlsx,.xls" onChange={e => handleFileUpload(e, 'unknown')} />
        </div>
        <div>
          <label className="block font-medium mb-2">Upload Actual Data (features + SepsisLabel):</label>
          <input type="file" accept=".csv,.xlsx,.xls" onChange={e => handleFileUpload(e, 'actual')} />
        </div>
      </div>
      {error && <div className="text-danger-600 font-medium mb-4">{error}</div>}
      {combined.length > 0 && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={combined} margin={{ top: 20, right: 30, left: 0, bottom: 40 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="index"
                label={{ value: 'Sample Index', position: 'insideBottom', offset: -10 }}
                angle={rotateLabels ? -45 : 0}
                textAnchor={rotateLabels ? 'end' : 'middle'}
                interval={rotateLabels ? 0 : 'preserveEnd'}
                height={rotateLabels ? 60 : 30}
              />
              <YAxis label={{ value: 'Sepsis Score', angle: -90, position: 'insideLeft' }} />
              <Tooltip formatter={(value: any, name: string) => [value, name === 'actual' ? 'Actual Sepsis' : 'Predicted Sepsis']} />
              <Legend />
              <Line type="monotone" dataKey="actual" stroke="#dc2626" strokeWidth={2} name="Actual Sepsis" dot={false} />
              <Line type="monotone" dataKey="predicted" stroke="#3b82f6" strokeWidth={2} name="Predicted Sepsis" dot={false} strokeDasharray="6 3" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
};

export default SepsisModelPerformance; 