import numpy as np
import pandas as pd
import time
import utilities
from tqdm import tqdm

# IMPORTANT: this variable is the last two digits of the EARLIEST YEAR of data
# that will be considered when finding candidate and committee IDs c- it cannot go before 1990 as data before then is
# not available
earliest = 1990

# These globals are very important and determine which rows of csvs are considered in the analysis -----------------------
# All spending types - this is just here for comparison
all_types = set(('10 ', '10J', '11 ', '11J', '12 ', '13 ', '15 ', '15C', '15E', '15J', '15Z',
'18G', '18H', '18J', '18K','18L,', '18U', '19', '19J', '20 ', '21Y', '22H', '22Y', '22Z', '23Y',
'24C', '24E', '24F', '24G', '24H', '24I', '24T', '24K', '24P', '24R', '24Z', '28L', '29 ',
'30 ', '30T', '30K', '30G', '30J', '30F', '31 ', '31T', '31K', '31G', '31J', '31F', '32', '32T', '32K', '32G', '32J', '32F',
'40 ', '40Y', '40T', '40Z', '41 ', '41Y', '41T', '41Z', '42 ', '42Y', '42T', '42Z'))


for_spending_types = set(('10 ', '10J', '11 ', '11J', '12 ', '13 ', '15 ', '15C', '15E', '15J', '15Z','16C', '16F', '16G', '16H',
'18G', '18H', '18J', '18K','18L,', '18U', '19', '19J', '20 ', '21Y', '22H', '22Y', '22Z', '23Y',
'24C', '24E', '24F', '24G', '24H', '24I', '24T', '24K', '24P', '24R', '24Z', '28L', '29 ',
'30 ', '30T', '30K', '30G', '30J', '30F', '31 ', '31T', '31K', '31G', '31J', '31F', '32', '32T', '32K', '32G', '32J', '32F',
'40 ', '40Y', '40T', '40Z', '41 ', '41Y', '41T', '41Z', '42 ', '42Y', '42T', '42Z')) #
against_spending_types = set(('24A','24N')) #

cand_expend_codes = set((
'C10', 'C30', 'R00', 'R10', 'R30', 'R35', 'R50'
)) # R20 and R40 are not counted becauase they're state/local party/cand contributions

# For future use, the loan types that we could include are right here
loans = ['16C', '16F', '16G', '16H']

# When aggregating all election spending as a sanity check, the program WON'T include these catcodes:
do_not_aggregate = ['Z1000', 'Z1100', 'Z1200', 'Z1300', 'Z1400',
'Z4100', 'Z4200', 'Z4300', 'Z4400', 'Z4500',
'Z5000', 'Z5100', 'Z5200', 'Z5300',
'Z9000', 'Z9100', 'Z9500', 'Z9600',
'Z9700', 'Z9800',
 'Z9999']

# Columns for CRP base files --------
cands_columns = ('Cycle', 'FECCandID', 'CID', 'FirstLastP', 'Party', 'DistIDRunFor', 'DistIDCurr', 'CurrCand',
                'CycleCand', 'CRPICO', 'RecipCode', 'NoPacs')
cmtes_columns = ('Cycle', 'CmteID', 'PACShort', 'Affiliate', 'Ultorg', 'RecipID', 'RecipCode', 'FECCandID', 'Party',
                'PrimCode', 'Source', 'Sensitive', 'Foreign', 'Active')
pacs_columns = ('Cycle', 'FECRecNo', 'PACID', 'CID', 'Amount', 'Date', 'RealCode', 'Type', 'DI','FECCandID')
otherpacs_columns = ('Cycle', 'FECRecNo', 'Filerid', 'DonorCmte', 'ContribLendTrans', 'City', 'State', 'Zip',
                    'FECOccEmp', 'Primcode', 'Date', 'Amount', 'RecipID', 'Party', 'Otherid', 'RecipCode',
                    'RecipPrimcode', 'Amend', 'Report', 'PG', 'Microfilm', 'Type', 'RealCode', 'Source')
indivs_columns = ('Cycle', 'FECTransID', 'ContribID', 'Contrib', 'RecipID', 'Orgname', 'UltOrg', 'RealCode', 'Date',
                  'Amount', 'Street', 'City', 'State', 'Zip', 'RecipCode', 'Type', 'CmteID', 'OtherID', 'Gender',
                  'Microfilm', 'Occupation', 'Employer', 'Source')

# Dialects --
# Kwarg for opening crp csvs
crp_csv_kwargs = {'sep':',', 'quotechar':'|', 'skipinitialspace':True, 'encoding':'Latin-1', 'header':None}

# Create dialects and filepaths ---------------------------------------------------------------------

# Paths for 527 files - these do not depend on the cycle
cmtes527path = "data/raw_crp_spending_data/527/cmtes527.txt"
expends527path = "data/raw_crp_spending_data/527/expends527.txt"
rcpts527path = "data/raw_crp_spending_data/527/rcpts527.txt"
catcodes_path = "data/raw_crp_spending_data/CRP Industry Codes - 12.11.17.csv"

class CRP_Data():

    def __init__(self, year):

        # Make sure year is the right year
        if int(year) > 2016 or int(year) < 1990 or int(year) % 2 == 1:
            raise ValueError("Raw CRP data is not available for {}".format(year))

        # Get paths for raw data, as well as headers
        self.year = year
        self.candspath = 'data/raw_crp_spending_data/cands' + str(year)[2:4] + '.txt'
        self.cmtespath = 'data/raw_crp_spending_data/cmtes' + str(year)[2:4] + '.txt'
        self.pacspath = 'data/raw_crp_spending_data/pacs' + str(year)[2:4] + '.txt'
        self.otherpacspath = 'data/raw_crp_spending_data/pac_other' + str(year)[2:4] + '.txt'
        self.indivspath = 'data/raw_crp_spending_data/indivs' + str(year)[2:4] + '.txt'
        self.time0 = time.time()

    # Read the files - the reason these have their own definitions is to allow for various bits of data processing later on.
    def read_cands(self):
        cands = pd.read_csv(self.candspath, **crp_csv_kwargs)
        cands.columns = cands_columns
        cands = cands.drop_duplicates(keep = 'first') # Only drop if the entire line is duplicated
        return cands

    def read_cmtes(self):
        cmtes = pd.read_csv(self.cmtespath, **crp_csv_kwargs)
        cmtes.columns = cmtes_columns
        cmtes = cmtes.drop_duplicates(keep = 'first') # Only drop if the entire line is duplicated
        return cmtes

    def read_pacs(self):
        pacs = pd.read_csv(self.pacspath, **crp_csv_kwargs)
        pacs.columns = pacs_columns
        pacs['RealCode'] = pacs['RealCode'].apply(lambda x: str(x).upper())
        return pacs

    def read_otherpacs(self):
        otherpacs = pd.read_csv(self.otherpacspath, **crp_csv_kwargs, dtype = {7: 'str'}) # Column 7 is the zip code
        otherpacs.columns = otherpacs_columns
        return otherpacs

    def read_indivs(self, **kwargs):
        """
        Note that in 2016, this csv has 20 million + rows, so it might be worth reading this in in chunks with the
        nrows and skiprows argument.
        """
        print('Reading individuals file for {} at time {}'.format(self.year, time.time() - self.time0))
        indivs = pd.read_csv(self.indivspath, **crp_csv_kwargs, **kwargs)
        print('Finished reading individuals file for {} at time {}'.format(self.year, time.time() - self.time0))
        indivs.columns = indivs_columns
        indivs['RealCode'] = indivs['RealCode'].apply(lambda x: str(x).upper())
        return indivs

    def tabulate_spending_totals(self, num_chunks = 10):

        # Get metadata --

        # Catcodes
        catcodes_metadata = pd.read_csv(catcodes_path)
        catcodes_list = catcodes_metadata['Catcode'].unique().tolist()

        # Candidates
        cands_data = self.read_cands()
        cands_list = cands_data['CID'].unique().tolist()

        # Cmtes
        cmtes_data = self.read_cmtes()
        cmtes_list = cmtes_data['CmteID'].unique().tolist()

        # Start by calculating the sum of spending going directly to candidates from pacs
        pacs_data = self.read_pacs()
        result = pacs_data.groupby(['CID', 'RealCode'])['Amount'].sum().unstack().fillna(0)
        print(result)

        # Now add individual data. Note that the NaNs induced in the .add method are a result of the fact that
        # the result is taking a cartesian product of the realcodes and the recipids, and if there's no data in
        # either of the initial dataframes which contains a specific combination, then it will appear na even with
        # fill_value set to 0.

        indivs_data_length = utilities.get_filelength(self.indivspath)
        chunk_size = indivs_data_length // num_chunks + 1
        for chunk_start in tqdm(np.arange(0, indivs_data_length, chunk_size)):
            indivs_data = self.read_indivs(nrows = chunk_size, skiprows = chunk_start)
            to_add = indivs_data.groupby(['RecipID', 'RealCode'])['Amount'].sum().unstack().fillna(0)
            result = result.add(to_add, fill_value = 0).fillna(0)




data = CRP_Data(2000)
data.tabulate_spending_totals()


# Generate paths for recipient outfiles --
def create_cand_spending_path(year):
    return 'Spending Outputs/Cands/' + str(year) + " Candidate Spending by Industry.txt"

def create_dist_spending_path(year):
    return 'Spending Outputs/Dists/' + str(year) + " District Spending by Industry.txt"

def create_party_spending_path(year):
    return 'Spending Outputs/Parties/' + str(year) + " Party Spending by Industry.txt"

