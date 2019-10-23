from consts import *
import pandas as pd


class User:
    user_files_path = r'C:\Users\Geco\PycharmProjects\K_Means\resources\DNS-REQ-RES'
    # A list mapping index numbers (0, 1, 2...14) to user file numbers.
    user_file_numbers = [292, 301, 303, 305, 306, 308, 316, 334, 341, 343, 348, 354, 372, 387, 392]


    def __init__(self, i: 'The user index'):
        self.i = User.user_file_numbers[i]  # User file index mapped from user_file_numbers.
        self.df = self.get_user_data_frame()  # User dataframe, filtered for junk urls.
        # A Dataframe indexed by domain name with columns:
        #   Count - The number of times the domain appears in the user data.
        #   Percentage - (Domain count) / (Total number of domains)
        self.unique_domains = User.get_unique_domains(self.df['dns.qry.name'])


    @staticmethod
    def get_unique_domains(series) -> 'A Dataframe of unique domain names, their count and frequency':
        """Create a Dataframe from the users csv 'dns.qry.name' column:
            Index - Domain name.
            Count - The number of times the domain appears in the user data.
            Percentage - (Domain count) / (Total number of domains)
        """

        # A Series of unique domain name and their count.
        df = pd.DataFrame(series.value_counts())
        # Rename 'dns.qry.name' column to 'Count'.
        df = df.rename(columns={'dns.qry.name': 'Count'})
        # A Series of domain name and their frequency - percentage from the total number of domains.
        percentage = df.apply(lambda x: x / df.sum()[0])
        # Insert the 'Percentage column to the Dataframe.
        df.insert(1, 'Percentage', percentage)
        # Sort the Dataframe by descending order.
        df = df.sort_values(by='Percentage', ascending=False)

        return df


    def get_user_data_frame(self) -> 'The user filtered dataframe':
        """Extract the user data from its csv file
            and filter garbage url: domain names that do not have a dot ('.')
            within them.
        """

        # Get the user data from the user csv file.
        user_path = User.user_files_path + f'\\dnsSummary_user{self.i}.pcap.csv'
        user_df = pd.read_csv(user_path, dtype='str')

        # Filter non responsive urls.
        user_df = user_df[user_df['dns.flags.response'].str.match(r'1')]

        # Filter garbage urls.
        user_df = user_df[user_df['dns.qry.name'].str.match(r'\w+(\.\w+)+')]

        return user_df
