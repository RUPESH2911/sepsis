import React, { useState } from 'react';
import { Search, User, Play, AlertTriangle } from 'lucide-react';

interface PatientSearchProps {
  patientIds: string[];
  onPatientSelect: (patientId: string) => void;
  onStartDiagnosis: (patientId: string) => void;
  selectedPatientId?: string;
}

export const PatientSearch: React.FC<PatientSearchProps> = ({
  patientIds,
  onPatientSelect,
  onStartDiagnosis,
  selectedPatientId
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [filteredPatients, setFilteredPatients] = useState<string[]>([]);
  const [showDropdown, setShowDropdown] = useState(false);

  const handleSearch = (term: string) => {
    setSearchTerm(term);
    if (term.trim()) {
      const filtered = patientIds.filter(id => 
        id.toLowerCase().includes(term.toLowerCase())
      );
      setFilteredPatients(filtered);
      setShowDropdown(true);
    } else {
      setFilteredPatients([]);
      setShowDropdown(false);
    }
  };

  const handlePatientSelect = (patientId: string) => {
    setSearchTerm(patientId);
    setShowDropdown(false);
    onPatientSelect(patientId);
  };

  const handleStartDiagnosis = () => {
    if (selectedPatientId) {
      onStartDiagnosis(selectedPatientId);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="flex items-center space-x-3 mb-4">
        <Search className="w-5 h-5 text-primary-600" />
        <h3 className="text-lg font-semibold text-gray-900">Patient Search & Diagnosis</h3>
      </div>

      <div className="relative">
        <div className="relative">
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => handleSearch(e.target.value)}
            placeholder="Search patient by ID..."
            className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
          />
          <Search className="absolute left-3 top-3.5 w-4 h-4 text-gray-400" />
        </div>

        {/* Dropdown */}
        {showDropdown && filteredPatients.length > 0 && (
          <div className="absolute z-10 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg max-h-60 overflow-y-auto">
            {filteredPatients.map((patientId) => (
              <button
                key={patientId}
                onClick={() => handlePatientSelect(patientId)}
                className="w-full px-4 py-3 text-left hover:bg-gray-50 flex items-center space-x-3 border-b border-gray-100 last:border-b-0"
              >
                <User className="w-4 h-4 text-gray-500" />
                <span className="font-medium text-gray-900">{patientId}</span>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Selected Patient Actions */}
      {selectedPatientId && (
        <div className="mt-6 p-4 bg-primary-50 rounded-lg">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <User className="w-5 h-5 text-primary-600" />
              <div>
                <div className="font-medium text-primary-900">Selected Patient</div>
                <div className="text-sm text-primary-700">{selectedPatientId}</div>
              </div>
            </div>
            <button
              onClick={handleStartDiagnosis}
              className="flex items-center space-x-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
            >
              <Play className="w-4 h-4" />
              <span>Start Diagnosis</span>
            </button>
          </div>
        </div>
      )}

      {/* No Results */}
      {showDropdown && filteredPatients.length === 0 && searchTerm.trim() && (
        <div className="absolute z-10 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg p-4">
          <div className="flex items-center space-x-2 text-gray-500">
            <AlertTriangle className="w-4 h-4" />
            <span>No patients found matching "{searchTerm}"</span>
          </div>
        </div>
      )}
    </div>
  );
};