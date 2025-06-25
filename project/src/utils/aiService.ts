// Simulated AI/LLM service for clinical insights
export class AIService {
  private static instance: AIService;
  
  static getInstance(): AIService {
    if (!AIService.instance) {
      AIService.instance = new AIService();
    }
    return AIService.instance;
  }

  async analyzePatientData(patientData: any): Promise<any> {
    // Simulate AI analysis with realistic medical insights
    await this.delay(1000 + Math.random() * 2000);
    
    const riskFactors = this.identifyRiskFactors(patientData);
    const recommendations = this.generateRecommendations(riskFactors, patientData);
    const clinicalInsights = this.generateClinicalInsights(patientData, riskFactors);
    
    return {
      riskAssessment: {
        overallRisk: this.calculateOverallRisk(riskFactors),
        riskFactors,
        confidence: 0.85 + Math.random() * 0.1
      },
      recommendations,
      insights: clinicalInsights,
      nextSteps: this.generateNextSteps(riskFactors, patientData)
    };
  }

  async generateDatasetInsights(dataset: any[]): Promise<any> {
    await this.delay(2000 + Math.random() * 3000);
    
    const patterns = this.identifyDataPatterns(dataset);
    const qualityAssessment = this.assessDataQuality(dataset);
    const modelRecommendations = this.generateModelRecommendations(patterns);
    
    return {
      patterns,
      qualityAssessment,
      modelRecommendations,
      statisticalSummary: this.generateStatisticalSummary(dataset)
    };
  }

  async generateClinicalRecommendations(patientData: any, prediction: any): Promise<string[]> {
    await this.delay(500 + Math.random() * 1000);
    
    const recommendations = [];
    
    if (prediction.probability > 0.8) {
      recommendations.push("🚨 URGENT: Initiate sepsis bundle protocol immediately");
      recommendations.push("💊 Consider broad-spectrum antibiotics within 1 hour");
      recommendations.push("🩸 Order blood cultures before antibiotic administration");
      recommendations.push("💧 Begin aggressive fluid resuscitation (30ml/kg crystalloid)");
    } else if (prediction.probability > 0.6) {
      recommendations.push("⚠️ Monitor closely for sepsis development");
      recommendations.push("🧪 Consider additional lab work: procalcitonin, CRP");
      recommendations.push("📊 Increase vital sign monitoring frequency");
      recommendations.push("👨‍⚕️ Consider infectious disease consultation");
    } else if (prediction.probability > 0.4) {
      recommendations.push("📈 Continue standard monitoring protocols");
      recommendations.push("🔍 Watch for trending vital sign changes");
      recommendations.push("📋 Document any new symptoms or clinical changes");
    }

    // Add specific recommendations based on abnormal values
    if (patientData.labs?.Lactate > 2.5) {
      recommendations.push("🔬 Elevated lactate detected - consider metabolic acidosis workup");
    }
    
    if (patientData.vitals?.Temp > 38.3 || patientData.vitals?.Temp < 36) {
      recommendations.push("🌡️ Temperature dysregulation - investigate infectious source");
    }
    
    if (patientData.labs?.WBC > 12 || patientData.labs?.WBC < 4) {
      recommendations.push("🦠 Abnormal WBC count - suggests possible infection");
    }

    return recommendations;
  }

  private identifyRiskFactors(patientData: any): string[] {
    const factors = [];
    
    if (patientData.vitals?.HR > 100) factors.push("Tachycardia");
    if (patientData.vitals?.Temp > 38.3) factors.push("Hyperthermia");
    if (patientData.vitals?.Temp < 36) factors.push("Hypothermia");
    if (patientData.vitals?.SBP < 90) factors.push("Hypotension");
    if (patientData.vitals?.Resp > 22) factors.push("Tachypnea");
    if (patientData.labs?.Lactate > 2.5) factors.push("Elevated Lactate");
    if (patientData.labs?.WBC > 12) factors.push("Leukocytosis");
    if (patientData.labs?.WBC < 4) factors.push("Leukopenia");
    if (patientData.labs?.Platelets < 150) factors.push("Thrombocytopenia");
    if (patientData.ICULOS > 48) factors.push("Prolonged ICU Stay");
    
    return factors;
  }

  private calculateOverallRisk(riskFactors: string[]): 'low' | 'moderate' | 'high' | 'critical' {
    const riskScore = riskFactors.length;
    if (riskScore >= 6) return 'critical';
    if (riskScore >= 4) return 'high';
    if (riskScore >= 2) return 'moderate';
    return 'low';
  }

  private generateRecommendations(riskFactors: string[], patientData: any): string[] {
    const recommendations = [];
    
    if (riskFactors.includes("Hypotension")) {
      recommendations.push("Consider vasopressor support");
      recommendations.push("Evaluate fluid responsiveness");
    }
    
    if (riskFactors.includes("Elevated Lactate")) {
      recommendations.push("Serial lactate monitoring");
      recommendations.push("Assess tissue perfusion");
    }
    
    if (riskFactors.includes("Leukocytosis") || riskFactors.includes("Leukopenia")) {
      recommendations.push("Investigate infectious source");
      recommendations.push("Consider antibiotic therapy");
    }
    
    return recommendations;
  }

  private generateClinicalInsights(patientData: any, riskFactors: string[]): string[] {
    const insights = [];
    
    insights.push(`Patient shows ${riskFactors.length} sepsis risk factors`);
    insights.push(`Current SOFA score components suggest ${this.calculateOverallRisk(riskFactors)} risk`);
    
    if (patientData.ICULOS > 24) {
      insights.push("Extended ICU stay increases infection risk");
    }
    
    if (patientData.Age > 65) {
      insights.push("Advanced age is associated with increased sepsis mortality");
    }
    
    return insights;
  }

  private generateNextSteps(riskFactors: string[], patientData: any): string[] {
    const steps = [];
    
    steps.push("Continue hourly vital sign monitoring");
    steps.push("Reassess in 2-4 hours or if clinical status changes");
    
    if (riskFactors.length > 3) {
      steps.push("Consider ICU consultation");
      steps.push("Prepare for potential rapid response activation");
    }
    
    return steps;
  }

  private identifyDataPatterns(dataset: any[]): any {
    return {
      sepsisPrevalence: (dataset.filter(row => row.SepsisLabel === 1).length / dataset.length * 100).toFixed(1),
      avgICUStay: (dataset.reduce((sum, row) => sum + (row.ICULOS || 0), 0) / dataset.length).toFixed(1),
      commonRiskFactors: ["Elevated Lactate", "Tachycardia", "Hypotension"],
      temporalPatterns: "Sepsis onset typically occurs after 12-24 hours in ICU"
    };
  }

  private assessDataQuality(dataset: any[]): any {
    const totalCells = dataset.length * Object.keys(dataset[0] || {}).length;
    const missingCells = dataset.reduce((count, row) => {
      return count + Object.values(row).filter(val => val === null || val === undefined || val === '').length;
    }, 0);
    
    return {
      completeness: ((totalCells - missingCells) / totalCells * 100).toFixed(1),
      rowCount: dataset.length,
      columnCount: Object.keys(dataset[0] || {}).length,
      missingValueRate: (missingCells / totalCells * 100).toFixed(1)
    };
  }

  private generateModelRecommendations(patterns: any): string[] {
    return [
      "Consider ensemble methods (Random Forest + XGBoost) for optimal performance",
      "Implement time-series features for temporal pattern recognition",
      "Use SMOTE for handling class imbalance in sepsis cases",
      "Apply feature selection to identify most predictive biomarkers",
      "Implement cross-validation with temporal splits to avoid data leakage"
    ];
  }

  private generateStatisticalSummary(dataset: any[]): any {
    return {
      patientCount: new Set(dataset.map(row => row.Patient_ID)).size,
      avgAge: (dataset.reduce((sum, row) => sum + (row.Age || 0), 0) / dataset.length).toFixed(1),
      genderDistribution: {
        male: dataset.filter(row => row.Gender === 'M').length,
        female: dataset.filter(row => row.Gender === 'F').length
      },
      avgHospitalStay: (dataset.reduce((sum, row) => sum + (row.HospAdmTime || 0), 0) / dataset.length).toFixed(1)
    };
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}