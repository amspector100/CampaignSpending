import numpy as np
import pandas as pd
import time
import utilities
from tqdm import tqdm

# IMPORTANT: this variable is the last two digits of the EARLIEST YEAR of data
# that will be considered when finding candidate and committee IDs c- it cannot go before 1990 as data before then is
# not available
testcmte = 'C00002907'
earliest = 1990

# These globals are very important and determine which rows of csvs are considered in the analysis -----------------------
# All spending types - this is just here for comparison
all_types = set(('10 ', '10J', '11 ', '11J', '12 ', '13 ', '15 ', '15C', '15E', '15J', '15Z',
'18G', '18H', '18J', '18K','18L,', '18U', '19', '19J', '20 ', '21Y', '22H', '22Y', '22Z', '23Y',
'24C', '24E', '24F', '24G', '24H', '24I', '24T', '24K', '24P', '24R', '24Z', '28L', '29 ',
'30 ', '30T', '30K', '30G', '30J', '30F', '31 ', '31T', '31K', '31G', '31J', '31F', '32', '32T', '32K', '32G', '32J', '32F',
'40 ', '40Y', '40T', '40Z', '41 ', '41Y', '41T', '41Z', '42 ', '42Y', '42T', '42Z'))


for_spending_types = set(('10 ', '10J', '11 ', '11J', '12 ', '13 ', '15 ', '15C', '15E', '15J', '15Z','16C', '16F', '16G', '16H',
'18G', '18H', '18J', '18K','18L,', '18U', '19', '19J', '21Y', '22H', '22Y', '22Z', '23Y',
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
cmtes527path = "data/raw/crp_spending_data/527/cmtes527.txt"
expends527path = "data/raw/crp_spending_data/527/expends527.txt"
rcpts527path = "data/raw/crp_spending_data/527/rcpts527.txt"
catcodes_path = "data/raw/crp_spending_data/CRP Industry Codes - 12.11.17.csv"

class CRP_Spending_Data():
    """
    A class which organizes all the CRP Spending data. Includes methods for calculating total spending.
    """

    def __init__(self, year):

        # Make sure year is the right year
        if int(year) > 2016 or int(year) < 1990 or int(year) % 2 == 1:
            raise ValueError("Raw CRP data is not available for {}".format(year))

        # Get paths for raw data, as well as headers
        self.year = year
        self.candspath = 'data/raw/crp_spending_data/cands' + str(year)[2:4] + '.txt'
        self.cmtespath = 'data/raw/crp_spending_data/cmtes' + str(year)[2:4] + '.txt'
        self.pacspath = 'data/raw/crp_spending_data/pacs' + str(year)[2:4] + '.txt'
        self.otherpacspath = 'data/raw/crp_spending_data/pac_other' + str(year)[2:4] + '.txt'
        self.indivspath = 'data/raw/crp_spending_data/indivs' + str(year)[2:4] + '.txt'
        self.time0 = time.time()

    # Read the files - the reason these have their own definitions is to allow for various bits of data processing later on.
    def read_cands(self, subset = None):
        cands = pd.read_csv(self.candspath, **crp_csv_kwargs)
        cands.columns = cands_columns
        cands = cands.drop_duplicates(subset = subset, keep = 'first') # Only drop if the entire line is duplicated
        return cands

    def read_cmtes(self, subset = None):
        cmtes = pd.read_csv(self.cmtespath, **crp_csv_kwargs)
        cmtes.columns = cmtes_columns
        cmtes = cmtes.drop_duplicates(subset = subset, keep = 'first') # Only drop if the entire line is duplicated
        return cmtes

    def read_pacs(self, valid_types = None):

        # Read in data/columns
        pacs = pd.read_csv(self.pacspath, **crp_csv_kwargs)
        pacs.columns = pacs_columns

        # Format realcode and perhaps restrict only to valid spending types
        pacs['RealCode'] = pacs['RealCode'].apply(lambda x: str(x).upper())

        if valid_types is not None:
            pacs = pacs.loc[pacs['Type'].isin(valid_types)]

        return pacs

    def read_otherpacs(self, valid_types = None, quietly = False):
        """
        Reads otherpacs data file. It will also parse whether the filer id references the donor or the candidate, and subsets
        to only include valid spending types.
        :return:
        """

        # Read initial data and subset according to types
        otherpacs = pd.read_csv(self.otherpacspath, dtype = {7: 'str'}, **crp_csv_kwargs) # Column 7 is the zip code
        otherpacs.columns = otherpacs_columns
        if valid_types is not None:
            otherpacs = otherpacs.loc[otherpacs['Type'].isin(valid_types)]

        # In this data set, the 'Filderid' may refer either to the recipient of funds or to the donor. This function
        # quickly parses whether the Filerid refers to the receipient or the donor. This involves reading in cand data
        # to help with the parsing.
        cands_data = self.read_cands(subset = 'CID').set_index('CID')
        cmtes_data = self.read_cmtes(subset = 'CmteID').set_index('CmteID')

        def filer_is_donor(row):

            filerid = row['Filerid']#.lower()
            recipid = row['RecipID']#.lower()
            otherid = row['Otherid']#.lower()

            # Return unknown for rows with missing data
            if np.any(pd.isnull((filerid, recipid, otherid))):
                return "Unknown"

            # Handle the obvious cases (where the recipid and the filer/otherid are identical)
            if recipid == filerid:
                return False
            if recipid == otherid:
                return True

            # If the recipid is cands_data, check whether the other id is the same as the recipid
            if recipid in cands_data.index:
                if cands_data.loc[recipid, 'FECCandID'] == otherid:
                    return True
            if otherid in cmtes_data.index:
                if cmtes_data.loc[otherid, 'RecipID'] == recipid:
                    return True
            if filerid in cmtes_data.index:
                if cmtes_data.loc[filerid, 'RecipID'] == recipid:
                    return False

            return "Unknown"

        otherpacs['filer_is_donor'] = otherpacs.apply(filer_is_donor, axis = 1)

        # Report how successful the function was. Usually this is around 1% and arises from missing data.
        if quietly is not True:
            unknowns = (otherpacs['filer_is_donor'].astype(str) == 'Unknown').values.sum()
            unknowns_amount = otherpacs.loc[otherpacs['filer_is_donor'].astype(str) == 'Unknown', 'Amount'].sum()
            percent_unknowns = 100*unknowns/otherpacs.shape[0]
            print('For otherpacs data, unable to parse whether the filer is the donor for {} rows, constituting {}% of the data and ${}'.format(unknowns, percent_unknowns, unknowns_amount))

        # Create a column for the donorid and return
        otherpacs = otherpacs.loc[otherpacs['filer_is_donor'].astype(str) != 'Unknown']
        otherpacs['DonorID'] = otherpacs.apply(lambda row: row['Filerid'] if row['filer_is_donor'] else row['Otherid'], axis = 1)
        otherpacs.sample(300, replace = True).to_csv('Exploration/otherpacs_test.csv')
        return otherpacs

    def read_indivs(self, valid_types = None, **kwargs):
        """
        Note that in 2016, this csv has 20 million + rows, so it might be worth reading this in in chunks with the
        nrows and skiprows argument.
        :param valid_types: A list of types to subset to. Defaults to None.
        """
        #print('Reading individuals file for {} at time {}'.format(self.year, time.time() - self.time0))
        indivs = pd.read_csv(self.indivspath, dtype = {6:'str', 13:'str', 15:'str', 17:'str', 19:'str'}, **crp_csv_kwargs, **kwargs)
        #print('Finished reading individuals file for {} at time {}'.format(self.year, time.time() - self.time0))
        indivs.columns = indivs_columns
        indivs['RealCode'] = indivs['RealCode'].apply(lambda x: str(x).upper())

        # Subset to valid types
        if valid_types is not None:
            indivs = indivs.loc[indivs['Type'].isin(valid_types)]

        return indivs

    def tabulate_spending_totals(self, valid_types = None, num_chunks = 10):

        # First, deal with the contributions of "zcode committees" - these are basically shell or nested committees that
        # themselves receive money from third parties and then transfer it to candidates. We will track which industries
        # give money to party committees and then track which candidates the committees give money to. --------------

        # Get the list of zcodes we'll track (we don't include Z4 and Z9, which refer to candidate or party cmtes)
        catcodes_metadata = pd.read_csv(catcodes_path)
        catcodes_list = catcodes_metadata['Catcode'].unique().tolist()
        party_codes = [catcode for catcode in catcodes_list if catcode[0:2] == 'Z5']

        # Get all the committees with this affiliated catcode
        cmtes_data = self.read_cmtes()
        party_cmtes = cmtes_data.loc[cmtes_data['PrimCode'].isin(party_codes), 'CmteID'].values

        # Now run through spending data. We will be keeping a running total of three things. (1) Simple spending refers
        # to spending that does not include zcode cmtes. (2) The money received by party cmtes (party_receipts)
        # (3) The money spent by party cmtes (party_spending). We'll add them all together at the end.

        # -- -- -- -- -- Work through the pac data -- -- -- -- --
        pacs_data = self.read_pacs(valid_types = valid_types)
        party_spending_indexes = pacs_data['PACID'].isin(party_cmtes)
        simple_pacs_data = pacs_data.loc[~party_spending_indexes]
        party_pacs_data = pacs_data.loc[party_spending_indexes]

        # Aggregate - CID refers to CRP Candidate ID
        simple_spending = simple_pacs_data.groupby(['CID', 'RealCode'])['Amount'].sum().unstack().fillna(0)
        party_spending = party_pacs_data.groupby(['PACID', 'CID'])['Amount'].sum().unstack().fillna(0)

        # -- -- -- -- -- Work through other pac data -- -- -- -- --

        # Read and subset
        otherpacs_data = self.read_otherpacs(valid_types = valid_types)
        party_spending_indexes = otherpacs_data['RealCode'].isin(party_codes)
        party_receiving_indexes = otherpacs_data['RecipID'].isin(party_cmtes)
        otherpacs_party_spending = otherpacs_data.loc[party_spending_indexes]
        otherpacs_party_receipts = otherpacs_data.loc[party_receiving_indexes]
        otherpacs_simple_spending = otherpacs_data.loc[(~party_spending_indexes) & (~party_receiving_indexes)]

        # Continue to aggregate
        otherpacs_party_spending = otherpacs_party_spending.groupby(['DonorID', 'RecipID'])['Amount'].sum().unstack().fillna(0)
        party_spending = party_spending.add(otherpacs_party_spending, fill_value = 0).fillna(0)
        party_receipts = otherpacs_party_receipts.groupby(['RecipID', 'RealCode'])['Amount'].sum().unstack().fillna(0)
        simple_spending_to_add = otherpacs_simple_spending.groupby(['RecipID', 'RealCode'])['Amount'].sum().unstack().fillna(0)
        simple_spending = simple_spending.add(simple_spending_to_add, fill_value = 0).fillna(0)

        # -- -- -- -- -- Work through individual data, chunking -- -- -- -- --
        indivs_data_length = utilities.get_filelength(self.indivspath)
        chunk_size = indivs_data_length // num_chunks + 1
        print('For {}, working through individual spending'.format(self.year))
        for chunk_start in tqdm(np.arange(0, indivs_data_length, chunk_size)):

            # Read and subset
            indivs_data = self.read_indivs(nrows = chunk_size, skiprows = chunk_start, valid_types = valid_types)
            party_receiving_indexes = indivs_data['RecipID'].isin(party_cmtes)
            simple_indivs_data = indivs_data.loc[~party_receiving_indexes]
            party_indivs_data = indivs_data.loc[party_receiving_indexes]

            # Group and sum - RecipID is the recipient ID, which may be for a committee or a candidate.
            # RealCode is equivalent to PrimCode.
            simple_spending_to_add = simple_indivs_data.groupby(['RecipID', 'RealCode'])['Amount'].sum().unstack().fillna(0)
            simple_spending = simple_spending.add(simple_spending_to_add, fill_value = 0).fillna(0)
            party_receipts_to_add = party_indivs_data.groupby(['RecipID', 'RealCode'])['Amount'].sum().unstack().fillna(0)
            party_receipts = party_receipts.add(party_receipts_to_add, fill_value = 0).fillna(0)

        # -- -- -- -- -- Aggregate all the spending, accounting for where party comittees got their money -- -- -- -- --

        # Party Cmtes
        normalized_party_receipts = party_receipts.divide(party_receipts.sum(axis = 1), axis = 0)
        extra_receipt_cmtes = [c for c in party_receipts.index if c not in party_spending.index]
        extra_spending_cmtes = [c for c in party_spending.index if c not in party_receipts.index]
        party_spending = party_spending.append(pd.DataFrame(0, index = extra_receipt_cmtes, columns = party_spending.columns)).sort_index().T
        normalized_party_receipts = normalized_party_receipts.append(pd.DataFrame(0, index = extra_spending_cmtes, columns = normalized_party_receipts.columns)).sort_index()
        final_party_contributions = party_spending.dot(normalized_party_receipts)
        final_spending = simple_spending.add(final_party_contributions, fill_value = 0).fillna(0)
        print(final_spending.values.sum())

        # In final spending, account for cmtes. After experimentation, I discovered that most of the cmtes which come from
        # party contributions result from inter-party transactions, i.e. transactions between Z5 or J1 cmtes. The cmtes
        # which come from simple contributions are mostly individuals donating to non-candidate pacs, i.e. an auto worker
        # donating to an auto worker labor union.

        # In practice, this means we just ignore all of the cmtes which are not associated with candidates.  Start by
        # identifying cand cmtes in final spending column
        cmtes_data = self.read_cmtes(subset='CmteID').set_index('CmteID')
        cand_cmtes = [c for c in final_spending.index if c in cmtes_data.index]
        cand_cmtes = [c for c in cand_cmtes if cmtes_data.loc[c, 'RecipID'][0] == 'N']

        # Convert cand_cmtes spending to cand ids; drop other cmtes, etc.
        spending_to_convert = final_spending.loc[cand_cmtes]
        spending_to_convert.index = spending_to_convert.index.map(lambda x: cmtes_data.loc[x, 'RecipID'])
        final_spending = final_spending.loc[(~final_spending.index.isin(cand_cmtes)) & (final_spending.index.to_series().apply(lambda x: str(x)[0] == 'N'))]
        final_spending = final_spending.add(spending_to_convert, fill_value = 0).fillna(0)

        # We're done! Cache and return :)
        return final_spending



if __name__ == '__main__':

    data = CRP_Spending_Data(2016)
    spending = data.tabulate_spending_totals(valid_types = for_spending_types, num_chunks=10)
    print(spending.values.sum())


# Generate paths for recipient outfiles --
def create_cand_spending_path(year):
    return 'Spending Outputs/Cands/' + str(year) + " Candidate Spending by Industry.txt"

def create_dist_spending_path(year):
    return 'Spending Outputs/Dists/' + str(year) + " District Spending by Industry.txt"

def create_party_spending_path(year):
    return 'Spending Outputs/Parties/' + str(year) + " Party Spending by Industry.txt"

