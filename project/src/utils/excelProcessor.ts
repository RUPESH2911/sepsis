import * as XLSX from 'xlsx';
import { DatasetRow, DatasetInfo } from '../types/dataset';
import Papa from 'papaparse';

export class ExcelProcessor {
  static async processExcelFile(file: File): Promise<{ data: DatasetRow[], info: DatasetInfo }> {
    return new Promise((resolve, reject) => {
      // CSV support
      if (file.name.toLowerCase().endsWith('.csv')) {
        Papa.parse(file, {
          header: true,
          skipEmptyLines: true,
          complete: (results) => {
            const data = (results.data as DatasetRow[]).map(row => {
              if (row.SepsisLabel !== undefined) {
                row.SepsisLabel = Number(row.SepsisLabel);
              }
              return row;
            });
            const headers = Object.keys(data[0] || {});
            const patientIds = [...new Set(data.map(row => String(row.Patient_ID || row.PatientID || row.patient_id || '')))].filter(id => id);
            const datasetInfo: DatasetInfo = {
              id: Date.now().toString(),
              name: file.name.replace(/\.(csv)$/i, ''),
              uploadDate: new Date(),
              size: file.size,
              rowCount: data.length,
              columnCount: headers.length,
              status: 'ready',
              features: headers,
              targetColumn: headers.find(h => h.toLowerCase().includes('sepsis')) || 'SepsisLabel',
              patientIds
            };
            resolve({ data, info: datasetInfo });
          },
          error: (error) => {
            reject(new Error('Failed to parse CSV: ' + error.message));
          }
        });
        return;
      }
      const reader = new FileReader();
      
      reader.onload = (e) => {
        try {
          const data = new Uint8Array(e.target?.result as ArrayBuffer);
          const workbook = XLSX.read(data, { type: 'array' });
          
          // Get the first worksheet
          const sheetName = workbook.SheetNames[0];
          const worksheet = workbook.Sheets[sheetName];
          
          // Convert to JSON
          const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 });
          
          if (jsonData.length < 2) {
            throw new Error('Excel file must contain at least a header row and one data row');
          }
          
          // Extract headers and data
          const headers = jsonData[0] as string[];
          const rows = jsonData.slice(1) as any[][];
          
          // Convert to DatasetRow format
          const processedData: DatasetRow[] = rows.map(row => {
            const rowData: DatasetRow = {};
            headers.forEach((header, index) => {
              rowData[header] = row[index] || '';
            });
            if (rowData.SepsisLabel !== undefined) {
              rowData.SepsisLabel = Number(rowData.SepsisLabel);
            }
            return rowData;
          });
          
          // Extract unique patient IDs
          const patientIds = [...new Set(processedData.map(row => String(row.Patient_ID || row.PatientID || row.patient_id || '')))].filter(id => id);
          
          const datasetInfo: DatasetInfo = {
            id: Date.now().toString(),
            name: file.name.replace(/\.(xlsx|xls)$/i, ''),
            uploadDate: new Date(),
            size: file.size,
            rowCount: processedData.length,
            columnCount: headers.length,
            status: 'ready',
            features: headers,
            targetColumn: headers.find(h => h.toLowerCase().includes('sepsis')) || 'SepsisLabel',
            patientIds
          };
          
          resolve({ data: processedData, info: datasetInfo });
        } catch (error) {
          reject(new Error(`Failed to process Excel file: ${(error as Error).message}`));
        }
      };
      
      reader.onerror = () => reject(new Error('Failed to read Excel file'));
      reader.readAsArrayBuffer(file);
    });
  }
  
  static validateDataset(data: DatasetRow[]): { isValid: boolean; errors: string[] } {
    const errors: string[] = [];
    
    if (data.length === 0) {
      errors.push('Dataset is empty');
      return { isValid: false, errors };
    }
    
    const requiredColumns = ['Patient_ID', 'Hour'];
    const firstRow = data[0];
    const columns = Object.keys(firstRow);
    
    // Check for required columns (flexible naming)
    const hasPatientId = columns.some(col => 
      col.toLowerCase().includes('patient') && col.toLowerCase().includes('id')
    );
    const hasHour = columns.some(col => 
      col.toLowerCase() === 'hour' || col.toLowerCase() === 'time'
    );
    
    if (!hasPatientId) {
      errors.push('Dataset must contain a Patient ID column');
    }
    
    if (!hasHour) {
      errors.push('Dataset must contain an Hour/Time column');
    }
    
    // Check for vital signs and lab values
    const vitalSigns = ['HR', 'Temp', 'SBP', 'DBP', 'Resp', 'O2Sat'];
    const labValues = ['WBC', 'Lactate', 'Creatinine', 'Platelets'];
    
    const hasVitals = vitalSigns.some(vital => 
      columns.some(col => col.toLowerCase().includes(vital.toLowerCase()))
    );
    const hasLabs = labValues.some(lab => 
      columns.some(col => col.toLowerCase().includes(lab.toLowerCase()))
    );
    
    if (!hasVitals) {
      errors.push('Dataset should contain vital signs data (HR, Temp, BP, etc.)');
    }
    
    if (!hasLabs) {
      errors.push('Dataset should contain laboratory values (WBC, Lactate, etc.)');
    }
    
    return { isValid: errors.length === 0, errors };
  }
}