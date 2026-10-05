# My-Anatomy-
{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "3d015d9a",
   "metadata": {},
   "source": [
    "# Mini Project — Exploratory Data Analysis on India's COVID-19 Data\n",
    "\n",
    "**Goal:** Uncover state-wise trends, recovery rates, and vaccination progress in India's COVID-19 pandemic, using nothing but the tools from Module 1: NumPy, Pandas, and Matplotlib/Seaborn.\n",
    "\n",
    "**Data sources** (both loaded live, no download needed):\n",
    "- **Case data:** `imdevskp/covid-19-india-data` — daily state-wise confirmed/recovered/death counts, sourced from MoHFW and covid19india.org (30 Jan 2020 – 6 Aug 2020).\n",
    "- **Vaccination data:** Our World in Data (OWID) — India's national vaccination totals (15 Jan 2021 onward).\n",
    "\n",
    "**Workflow:** Load → Inspect → Clean → Explore (groupby/pivot) → Visualise (12 charts) → Summarise findings.\n",
    "\n",
    "> **Note on the data window:** the case dataset covers the first wave of the pandemic in India (through Aug 2020), before vaccines existed. The vaccination dataset picks up separately from Jan 2021. We treat these as two connected parts of the same story: how the outbreak spread state by state, and how the vaccination rollout progressed afterward.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c220e20f",
   "metadata": {},
   "source": [
    "## 1. Setup"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b06d4494",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-08-21T02:52:17.544495Z",
     "iopub.status.busy": "2026-08-21T02:52:17.544348Z",
     "iopub.status.idle": "2026-08-21T02:52:18.855656Z",
     "shell.execute_reply": "2026-08-21T02:52:18.854539Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Ready.\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "sns.set_theme(style=\"whitegrid\")\n",
    "plt.rcParams[\"figure.dpi\"] = 110\n",
    "pd.set_option(\"display.max_columns\", 20)\n",
    "\n",
    "print(\"Ready.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3b796397",
   "metadata": {},
   "source": [
    "## 2. Load the Data\n",
    "\n",
    "We read both CSVs straight from their public GitHub URLs with `pd.read_csv()` — exactly the File I/O skill from Module 1."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3e4f569d",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-08-21T02:52:18.858203Z",
     "iopub.status.busy": "2026-08-21T02:52:18.857356Z",
     "iopub.status.idle": "2026-08-21T02:52:19.262875Z",
     "shell.execute_reply": "2026-08-21T02:52:19.261774Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Cases shape: (4692, 10)\n",
      "Vaccination shape: (1250, 8)\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Date</th>\n",
       "      <th>Name of State / UT</th>\n",
       "      <th>Latitude</th>\n",
       "      <th>Longitude</th>\n",
       "      <th>Total Confirmed cases</th>\n",
       "      <th>Death</th>\n",
       "      <th>Cured/Discharged/Migrated</th>\n",
       "      <th>New cases</th>\n",
       "      <th>New deaths</th>\n",
       "      <th>New recovered</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2020-01-30</td>\n",
       "      <td>Kerala</td>\n",
       "      <td>10.8505</td>\n",
       "      <td>76.2711</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2020-01-31</td>\n",
       "      <td>Kerala</td>\n",
       "      <td>10.8505</td>\n",
       "      <td>76.2711</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2020-02-01</td>\n",
       "      <td>Kerala</td>\n",
       "      <td>10.8505</td>\n",
       "      <td>76.2711</td>\n",
       "      <td>2.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2020-02-02</td>\n",
       "      <td>Kerala</td>\n",
       "      <td>10.8505</td>\n",
       "      <td>76.2711</td>\n",
       "      <td>3.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2020-02-03</td>\n",
       "      <td>Kerala</td>\n",
       "      <td>10.8505</td>\n",
       "      <td>76.2711</td>\n",
       "      <td>3.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         Date Name of State / UT  Latitude  Longitude  Total Confirmed cases  \\\n",
       "0  2020-01-30             Kerala   10.8505    76.2711                    1.0   \n",
       "1  2020-01-31             Kerala   10.8505    76.2711                    1.0   \n",
       "2  2020-02-01             Kerala   10.8505    76.2711                    2.0   \n",
       "3  2020-02-02             Kerala   10.8505    76.2711                    3.0   \n",
       "4  2020-02-03             Kerala   10.8505    76.2711                    3.0   \n",
       "\n",
       "  Death  Cured/Discharged/Migrated  New cases  New deaths  New recovered  \n",
       "0     0                        0.0          0           0              0  \n",
       "1     0                        0.0          0           0              0  \n",
       "2     0                        0.0          1           0              0  \n",
       "3     0                        0.0          1           0              0  \n",
       "4     0                        0.0          0           0              0  "
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "cases = pd.read_csv(\n",
    "    \"https://raw.githubusercontent.com/imdevskp/covid-19-india-data/master/complete.csv\"\n",
    ")\n",
    "vax = pd.read_csv(\n",
    "    \"https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/country_data/India.csv\"\n",
    ")\n",
    "\n",
    "print(\"Cases shape:\", cases.shape)\n",
    "print(\"Vaccination shape:\", vax.shape)\n",
    "cases.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8dc525ff",
   "metadata": {},
   "outputs": [],
   "source": [
    "vax.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a2a024a5",
   "metadata": {},
   "source": [
    "## 3. Inspect Before Touching Anything\n",
    "\n",
    "Always check shape, dtypes, and missing values first — this tells you exactly what cleaning is needed."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5889ea3f",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-08-21T02:52:19.265346Z",
     "iopub.status.busy": "2026-08-21T02:52:19.264636Z",
     "iopub.status.idle": "2026-08-21T02:52:19.270996Z",
     "shell.execute_reply": "2026-08-21T02:52:19.269929Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Date                             str\n",
       "Name of State / UT               str\n",
       "Latitude                     float64\n",
       "Longitude                    float64\n",
       "Total Confirmed cases        float64\n",
       "Death                            str\n",
       "Cured/Discharged/Migrated    float64\n",
       "New cases                      int64\n",
       "New deaths                     int64\n",
       "New recovered                  int64\n",
       "dtype: object"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "cases.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "492c7936",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-08-21T02:52:19.272801Z",
     "iopub.status.busy": "2026-08-21T02:52:19.272596Z",
     "iopub.status.idle": "2026-08-21T02:52:19.279542Z",
     "shell.execute_reply": "2026-08-21T02:52:19.278786Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Date                         0\n",
       "Name of State / UT           0\n",
       "Latitude                     0\n",
       "Longitude                    0\n",
       "Total Confirmed cases        0\n",
       "Death                        0\n",
       "Cured/Discharged/Migrated    0\n",
       "New cases                    0\n",
       "New deaths                   0\n",
       "New recovered                0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "cases.isna().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f2e8cf81",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-08-21T02:52:19.281064Z",
     "iopub.status.busy": "2026-08-21T02:52:19.280918Z",
     "iopub.status.idle": "2026-08-21T02:52:19.287432Z",
     "shell.execute_reply": "2026-08-21T02:52:19.286552Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of unique state/UT names: 40\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "['Andaman and Nicobar Islands',\n",
       " 'Andhra Pradesh',\n",
       " 'Arunachal Pradesh',\n",
       " 'Assam',\n",
       " 'Bihar',\n",
       " 'Chandigarh',\n",
       " 'Chhattisgarh',\n",
       " 'Dadra and Nagar Haveli and Daman and Diu',\n",
       " 'Delhi',\n",
       " 'Goa',\n",
       " 'Gujarat',\n",
       " 'Haryana',\n",
       " 'Himachal Pradesh',\n",
       " 'Jammu and Kashmir',\n",
       " 'Jharkhand',\n",
       " 'Karnataka',\n",
       " 'Kerala',\n",
       " 'Ladakh',\n",
       " 'Madhya Pradesh',\n",
       " 'Maharashtra',\n",
       " 'Manipur',\n",
       " 'Meghalaya',\n",
       " 'Mizoram',\n",
       " 'Nagaland',\n",
       " 'Odisha',\n",
       " 'Puducherry',\n",
       " 'Punjab',\n",
       " 'Rajasthan',\n",
       " 'Sikkim',\n",
       " 'Tamil Nadu',\n",
       " 'Telangana',\n",
       " 'Telangana***',\n",
       " 'Telengana',\n",
       " 'Tripura',\n",
       " 'Union Territory of Chandigarh',\n",
       " 'Union Territory of Jammu and Kashmir',\n",
       " 'Union Territory of Ladakh',\n",
       " 'Uttar Pradesh',\n",
       " 'Uttarakhand',\n",
       " 'West Bengal']"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "print(\"Number of unique state/UT names:\", cases[\"Name of State / UT\"].nunique())\n",
    "sorted(cases[\"Name of State / UT\"].unique())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "08979bae",
   "metadata": {},
   "source": [
    "Two problems jump out immediately:\n",
    "\n",
    "1. **`Death` is stored as text (object), not a number** — likely because of stray characters somewhere in the column.\n",
    "2. **The same state appears under multiple spellings**: `Telangana`, `Telengana`, and `Telangana***` are all the same state. Similarly, a few Union Territories are listed twice — once plainly (`Ladakh`) and once with a `\"Union Territory of ...\"` prefix.\n",
    "\n",
    "This is exactly the kind of real-world messiness the handbook's Data Leakage/cleaning ideas prepare you for — we fix it before doing any analysis, never after."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d7b4c311",
   "metadata": {},
   "source": [
    "## 4. Clean the Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f57ef862",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-08-21T02:52:19.290218Z",
     "iopub.status.busy": "2026-08-21T02:52:19.289461Z",
     "iopub.status.idle": "2026-08-21T02:52:19.312288Z",
     "shell.execute_reply": "2026-08-21T02:52:19.311274Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Clean state count: 35\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "date             datetime64[us]\n",
       "state                       str\n",
       "lat                     float64\n",
       "long                    float64\n",
       "confirmed                 int64\n",
       "deaths                    int64\n",
       "cured                     int64\n",
       "new_cases                 int64\n",
       "new_deaths                int64\n",
       "new_recovered             int64\n",
       "active                    int64\n",
       "dtype: object"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Rename to clean, snake_case column names\n",
    "cases.columns = [\n",
    "    \"date\", \"state\", \"lat\", \"long\", \"confirmed\",\n",
    "    \"deaths\", \"cured\", \"new_cases\", \"new_deaths\", \"new_recovered\",\n",
    "]\n",
    "\n",
    "# Fix data types\n",
    "cases[\"date\"] = pd.to_datetime(cases[\"date\"])\n",
    "cases[\"deaths\"] = pd.to_numeric(cases[\"deaths\"], errors=\"coerce\").fillna(0).astype(int)\n",
    "cases[\"confirmed\"] = cases[\"confirmed\"].astype(int)\n",
    "cases[\"cured\"] = cases[\"cured\"].astype(int)\n",
    "\n",
    "# Fix inconsistent state names\n",
    "name_fix = {\n",
    "    \"Telangana***\": \"Telangana\",\n",
    "    \"Telengana\": \"Telangana\",\n",
    "    \"Union Territory of Jammu and Kashmir\": \"Jammu and Kashmir\",\n",
    "    \"Union Territory of Ladakh\": \"Ladakh\",\n",
    "    \"Union Territory of Chandigarh\": \"Chandigarh\",\n",
    "}\n",
    "cases[\"state\"] = cases[\"state\"].replace(name_fix)\n",
    "\n",
    "# Drop any exact duplicate (date, state) rows this merge might create\n",
    "cases = cases.drop_duplicates(subset=[\"date\", \"state\"]).sort_values([\"state\", \"date\"]).reset_index(drop=True)\n",
    "\n",
    "# A derived column: active cases = confirmed - deaths - recovered\n",
    "cases[\"active\"] = cases[\"confirmed\"] - cases[\"deaths\"] - cases[\"cured\"]\n",
    "\n",
    "print(\"Clean state count:\", cases[\"state\"].nunique())\n",
    "cases.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f0599487",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-08-21T02:52:19.314392Z",
     "iopub.status.busy": "2026-08-21T02:52:19.313813Z",
     "iopub.status.idle": "2026-08-21T02:52:19.319280Z",
     "shell.execute_reply": "2026-08-21T02:52:19.318482Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "np.int64(0)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "cases.isna().sum().sum()   # should be 0 -- fully clean now"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ebee73a5",
   "metadata": {},
   "source": [
    "## 5. National-Level Summary (groupby)\n",
    "\n",
    "Let's aggregate all states together, per date, using `groupby` — the Module 1 workhorse."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "30616b7b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-08-21T02:52:19.321282Z",
     "iopub.status.busy": "2026-08-21T02:52:19.320634Z",
     "iopub.status.idle": "2026-08-21T02:52:19.334714Z",
     "shell.execute_reply": "2026-08-21T02:52:19.333845Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>date</th>\n",
       "      <th>confirmed</th>\n",
       "      <th>deaths</th>\n",
       "      <th>cured</th>\n",
       "      <th>recovery_rate</th>\n",
       "      <th>death_rate</th>\n",
       "      <th>new_confirmed</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>181</th>\n",
       "      <td>2020-08-02</td>\n",
       "      <td>1750723</td>\n",
       "      <td>37364</td>\n",
       "      <td>1145629</td>\n",
       "      <td>65.44</td>\n",
       "      <td>2.13</td>\n",
       "      <td>54735.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>182</th>\n",
       "      <td>2020-08-03</td>\n",
       "      <td>1803695</td>\n",
       "      <td>38135</td>\n",
       "      <td>1186203</td>\n",
       "      <td>65.77</td>\n",
       "      <td>2.11</td>\n",
       "      <td>52972.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>183</th>\n",
       "      <td>2020-08-04</td>\n",
       "      <td>1855745</td>\n",
       "      <td>38938</td>\n",
       "      <td>1230509</td>\n",
       "      <td>66.31</td>\n",
       "      <td>2.10</td>\n",
       "      <td>52050.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>184</th>\n",
       "      <td>2020-08-05</td>\n",
       "      <td>1908254</td>\n",
       "      <td>39795</td>\n",
       "      <td>1282215</td>\n",
       "      <td>67.19</td>\n",
       "      <td>2.09</td>\n",
       "      <td>52509.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>185</th>\n",
       "      <td>2020-08-06</td>\n",
       "      <td>1964536</td>\n",
       "      <td>40699</td>\n",
       "      <td>1328336</td>\n",
       "      <td>67.62</td>\n",
       "      <td>2.07</td>\n",
       "      <td>56282.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          date  confirmed  deaths    cured  recovery_rate  death_rate  \\\n",
       "181 2020-08-02    1750723   37364  1145629          65.44        2.13   \n",
       "182 2020-08-03    1803695   38135  1186203          65.77        2.11   \n",
       "183 2020-08-04    1855745   38938  1230509          66.31        2.10   \n",
       "18
