import pandas as pd
from scipy.stats import f_oneway, ttest_ind


class ABHypothesisTesting:
    def __init__(self, data):
        self.data = data

    def test_risk_across_provinces(self):
        province_groups = [self.data[self.data['Province'] == p]['TotalPremium'].dropna() for p in self.data['Province'].unique()]
        province_groups = [group for group in province_groups if len(group) > 1]
        if len(province_groups) < 2:
            return {
                "Test": "ANOVA",
                "Null Hypothesis": "No risk differences between postal codes",
                "F-Statistic": None,
                "p-Value": None,
                "Reject Null": False,
                "Error": "Not enough valid groups for ANOVA."
            }
        # Check variance
        if any(group.var() == 0 for group in province_groups):
            return {
                "Test": "ANOVA",
                "Null Hypothesis": "No risk differences between postal codes",
                "F-Statistic": None,
                "p-Value": None,
                "Reject Null": False,
                "Error": "One or more groups have zero variance."
            }
        # Perform ANOVA
        result = f_oneway(*province_groups)
        return {
            "Test": "ANOVA",
            "Null Hypothesis": "No risk differences between postal codes",
            "F-Statistic": result.statistic,
            "p-Value": result.pvalue,
            "Reject Null": result.pvalue < 0.05
        }

    def test_risk_between_PostalCode(self):
        # Group data by postal code
        grouped_data = [self.data[self.data['PostalCode'] == p]['TotalPremium'].dropna() for p in self.data['PostalCode'].unique()]
        grouped_data = [group for group in grouped_data if len(group) > 1]  # Keep only groups with sufficient data
        if len(grouped_data) < 2:
            return {
                "Test": "ANOVA",
                "Null Hypothesis": "No risk differences between postal codes",
                "F-Statistic": None,
                "p-Value": None,
                "Reject Null": False,
                "Error": "Not enough valid groups for ANOVA."
            }
        # Check variance
        if any(group.var() == 0 for group in grouped_data):
            return {
                "Test": "ANOVA",
                "Null Hypothesis": "No risk differences between postal codes",
                "F-Statistic": None,
                "p-Value": None,
                "Reject Null": False,
                "Error": "One or more groups have zero variance."
            }
        # Perform ANOVA
        result = f_oneway(*grouped_data)
        return {
            "Test": "ANOVA",
            "Null Hypothesis": "No risk differences between postal codes",
            "F-Statistic": result.statistic,
            "p-Value": result.pvalue,
            "Reject Null": result.pvalue < 0.05
        }

    def test_margin_difference_between_PostalCode(self):
        """
        Test if there are significant margin (profit) differences between zip codes.
        Null Hypothesis: There are no significant margin differences between zip codes.
        """
        self.data['Margin'] = self.data['TotalPremium'] - self.data['TotalClaims']
        postalcode_groups = [self.data[self.data['PostalCode'] == z]['Margin'] for z in self.data['PostalCode'].unique()]
        result = f_oneway(*postalcode_groups)
        return {
            "Test": "ANOVA",
            "Null Hypothesis": "No significant margin differences between postal codes",
            "F-Statistic": result.statistic,
            "p-Value": result.pvalue,
            "Reject Null": result.pvalue < 0.05
        }

    def test_risk_difference_gender(self):
        """
        Test if there are significant risk differences (Total Claims) between genders.
        Null Hypothesis: There are no significant risk differences between women and men.
        """
        male_group = self.data[self.data['Gender'] == 'Male']['TotalPremium']
        female_group = self.data[self.data['Gender'] == 'Female']['TotalPremium']
        result = ttest_ind(male_group, female_group, equal_var=False)
        return {
            "Test": "T-Test",
            "Null Hypothesis": "No significant risk differences between women and men",
            "T-Statistic": result.statistic,
            "p-Value": result.pvalue,
            "Reject Null": result.pvalue < 0.05
        }
