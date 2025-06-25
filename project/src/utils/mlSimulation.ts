import { PatientData, PredictionResult } from '../types/patient';

export const simulateMLPrediction = (patient: PatientData): PredictionResult => {
  // Simulate ML model prediction based on key sepsis indicators
  let riskScore = 0;
  
  // Vital signs risk factors
  if (patient.vitals.HR > 100) riskScore += 0.15;
  if (patient.vitals.Temp > 38.3 || patient.vitals.Temp < 36) riskScore += 0.2;
  if (patient.vitals.SBP < 90) riskScore += 0.25;
  if (patient.vitals.Resp > 22) riskScore += 0.15;
  if (patient.vitals.O2Sat < 95) riskScore += 0.1;
  
  // Lab values risk factors
  if (patient.labs.WBC > 12 || patient.labs.WBC < 4) riskScore += 0.2;
  if (patient.labs.Lactate > 2.5) riskScore += 0.3;
  if (patient.labs.Creatinine > 1.5) riskScore += 0.15;
  if (patient.labs.Bilirubin_total > 2.0) riskScore += 0.1;
  if (patient.labs.Platelets < 150) riskScore += 0.15;
  
  // Time-based factors
  if (patient.ICULOS > 12) riskScore += 0.1;
  
  // Add some noise to make it more realistic
  riskScore += (Math.random() - 0.5) * 0.2;
  riskScore = Math.max(0, Math.min(1, riskScore));
  
  // Determine risk level
  let riskLevel: 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL';
  if (riskScore < 0.3) riskLevel = 'LOW';
  else if (riskScore < 0.6) riskLevel = 'MODERATE';
  else if (riskScore < 0.8) riskLevel = 'HIGH';
  else riskLevel = 'CRITICAL';
  
  // Calculate confidence (higher confidence for extreme values)
  const confidence = Math.min(0.95, 0.6 + Math.abs(riskScore - 0.5) * 0.7);
  
  // Estimate time to sepsis onset if high risk
  let timeToSepsis: number | undefined;
  if (riskScore > 0.6) {
    timeToSepsis = Math.max(1, Math.round((1 - riskScore) * 12)); // 1-12 hours
  }
  
  return {
    probability: riskScore,
    confidence,
    riskLevel,
    timeToSepsis
  };
};

export const calculateTimeSavings = (
  patientTimeline: PatientData[]
): { actualOnsetHour: number; predictedOnsetHour: number; timeSaved: number } | null => {
  // Find actual sepsis onset
  const actualOnsetHour = patientTimeline.findIndex(p => p.SepsisLabel === 1);
  if (actualOnsetHour === -1) return null;
  
  // Find when model would have predicted with high confidence (>0.8)
  let predictedOnsetHour = -1;
  for (let i = 0; i < actualOnsetHour; i++) {
    const prediction = simulateMLPrediction(patientTimeline[i]);
    if (prediction.probability > 0.8) {
      predictedOnsetHour = i;
      break;
    }
  }
  
  if (predictedOnsetHour === -1) return null;
  
  return {
    actualOnsetHour: actualOnsetHour + 1, // Convert to 1-based
    predictedOnsetHour: predictedOnsetHour + 1,
    timeSaved: actualOnsetHour - predictedOnsetHour
  };
};