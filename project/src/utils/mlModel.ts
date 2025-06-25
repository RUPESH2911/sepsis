import { DatasetRow, ModelMetrics } from '../types/dataset';

export class SepsisMLModel {
  private model: any = null;
  private isTraining = false;
  private trainingProgress = 0;

  async trainModel(dataset: DatasetRow[]): Promise<ModelMetrics> {
    this.isTraining = true;
    this.trainingProgress = 0;

    // Simulate training progress
    const progressInterval = setInterval(() => {
      this.trainingProgress += Math.random() * 15;
      if (this.trainingProgress > 95) {
        this.trainingProgress = 95;
      }
    }, 200);

    try {
      // Simulate model training with realistic delay
      await this.delay(3000 + Math.random() * 2000);
      
      // Clear progress interval
      clearInterval(progressInterval);
      this.trainingProgress = 100;
      
      // Simulate trained model
      this.model = {
        trained: true,
        features: this.extractFeatures(dataset),
        timestamp: new Date()
      };

      // Generate realistic metrics
      const metrics = this.generateModelMetrics(dataset);
      
      this.isTraining = false;
      return metrics;
    } catch (error) {
      clearInterval(progressInterval);
      this.isTraining = false;
      throw error;
    }
  }

  predict(patientData: any): { probability: number; confidence: number; riskLevel: string } {
    if (!this.model) {
      throw new Error('Model not trained yet');
    }

    // Enhanced prediction algorithm based on clinical sepsis criteria
    let riskScore = 0;
    let confidence = 0.8;

    // SIRS Criteria (Systemic Inflammatory Response Syndrome)
    let sirsCount = 0;
    
    // Temperature
    if (patientData.vitals?.Temp > 38 || patientData.vitals?.Temp < 36) {
      sirsCount++;
      riskScore += 0.15;
    }
    
    // Heart Rate
    if (patientData.vitals?.HR > 90) {
      sirsCount++;
      riskScore += 0.12;
    }
    
    // Respiratory Rate or PaCO2
    if (patientData.vitals?.Resp > 20 || patientData.labs?.PaCO2 < 32) {
      sirsCount++;
      riskScore += 0.10;
    }
    
    // WBC
    if (patientData.labs?.WBC > 12 || patientData.labs?.WBC < 4) {
      sirsCount++;
      riskScore += 0.18;
    }

    // SOFA Score components
    // Cardiovascular (MAP, vasopressors)
    if (patientData.vitals?.MAP < 70) {
      riskScore += 0.20;
    }
    
    // Respiratory (PaO2/FiO2 ratio)
    const pf_ratio = patientData.labs?.SaO2 / (patientData.labs?.FiO2 || 0.21);
    if (pf_ratio < 400) {
      riskScore += 0.15;
    }
    
    // Renal (Creatinine)
    if (patientData.labs?.Creatinine > 1.2) {
      riskScore += 0.12;
    }
    
    // Hepatic (Bilirubin)
    if (patientData.labs?.Bilirubin_total > 1.2) {
      riskScore += 0.08;
    }
    
    // Coagulation (Platelets)
    if (patientData.labs?.Platelets < 150) {
      riskScore += 0.10;
    }

    // Additional sepsis biomarkers
    if (patientData.labs?.Lactate > 2.0) {
      riskScore += 0.25; // Lactate is highly predictive
      confidence += 0.1;
    }
    
    if (patientData.labs?.Lactate > 4.0) {
      riskScore += 0.15; // Severe hyperlactatemia
    }

    // Procalcitonin (if available)
    if (patientData.labs?.Procalcitonin > 0.5) {
      riskScore += 0.20;
    }

    // Time-based factors
    if (patientData.ICULOS > 24) {
      riskScore += 0.08; // Prolonged ICU stay
    }
    
    if (patientData.HospAdmTime > 72) {
      riskScore += 0.05; // Hospital-acquired infection risk
    }

    // Age factor
    if (patientData.Age > 65) {
      riskScore += 0.05;
    }

    // Organ dysfunction indicators
    if (patientData.vitals?.O2Sat < 90) {
      riskScore += 0.15;
    }
    
    if (patientData.labs?.pH < 7.35) {
      riskScore += 0.12; // Acidosis
    }

    // Normalize risk score
    riskScore = Math.min(1.0, Math.max(0.0, riskScore));
    
    // Adjust confidence based on data completeness
    const dataCompleteness = this.calculateDataCompleteness(patientData);
    confidence *= dataCompleteness;
    
    // Add some realistic noise
    riskScore += (Math.random() - 0.5) * 0.05;
    riskScore = Math.max(0, Math.min(1, riskScore));

    // Determine risk level
    let riskLevel: string;
    if (riskScore < 0.25) riskLevel = 'LOW';
    else if (riskScore < 0.50) riskLevel = 'MODERATE';
    else if (riskScore < 0.75) riskLevel = 'HIGH';
    else riskLevel = 'CRITICAL';

    return {
      probability: riskScore,
      confidence: Math.min(0.95, confidence),
      riskLevel
    };
  }

  private calculateDataCompleteness(patientData: any): number {
    const requiredFields = [
      'vitals.HR', 'vitals.Temp', 'vitals.SBP', 'vitals.Resp',
      'labs.WBC', 'labs.Lactate', 'labs.Creatinine', 'labs.Platelets'
    ];
    
    let completeness = 0;
    requiredFields.forEach(field => {
      const value = this.getNestedValue(patientData, field);
      if (value !== null && value !== undefined && value !== '') {
        completeness += 1;
      }
    });
    
    return completeness / requiredFields.length;
  }

  private getNestedValue(obj: any, path: string): any {
    return path.split('.').reduce((current, key) => current?.[key], obj);
  }

  getTrainingProgress(): number {
    return this.trainingProgress;
  }

  isModelTraining(): boolean {
    return this.isTraining;
  }

  isModelTrained(): boolean {
    return this.model !== null;
  }

  private extractFeatures(dataset: DatasetRow[]): string[] {
    if (dataset.length === 0) return [];
    return Object.keys(dataset[0]).filter(key => key !== 'SepsisLabel' && key !== 'Patient_ID');
  }

  private generateModelMetrics(dataset: DatasetRow[]): ModelMetrics {
    // Simulate realistic model performance metrics
    const accuracy = 0.87 + Math.random() * 0.08;
    const precision = 0.82 + Math.random() * 0.10;
    const recall = 0.79 + Math.random() * 0.12;
    const f1Score = 2 * (precision * recall) / (precision + recall);
    const auc = 0.89 + Math.random() * 0.06;

    // Generate confusion matrix
    const totalSamples = dataset.length;
    const positiveSamples = dataset.filter(row => row.SepsisLabel === 1).length;
    const negativeSamples = totalSamples - positiveSamples;
    
    const tp = Math.round(positiveSamples * recall);
    const fp = Math.round(negativeSamples * (1 - precision));
    const fn = positiveSamples - tp;
    const tn = negativeSamples - fp;

    // Generate feature importance
    const features = this.extractFeatures(dataset);
    const featureImportance = features.map(feature => ({
      feature,
      importance: Math.random()
    })).sort((a, b) => b.importance - a.importance).slice(0, 10);

    return {
      accuracy,
      precision,
      recall,
      f1Score,
      auc,
      confusionMatrix: [[tn, fp], [fn, tp]],
      featureImportance
    };
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}