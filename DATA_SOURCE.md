# Data Source & Verification

## Source
- **Dataset:** Medical Appointment No Shows
- **Author:** Joni Hoppen / Aquarela Analytics
- **URL:** https://www.kaggle.com/datasets/joniarroba/noshowappointments
- **Version:** 5
- **Licence:** CC BY-NC-SA 4.0 (attribution required; non-commercial; share-alike)
- **File:** KaggleV2-May-2016.csv (10,739,535 bytes; file timestamp 2019-09-20)
- **Downloaded:** 2026-09-29 by Dave

## Integrity
- **SHA-256:** 9132d3e7d0246617df9041d3764f20ad6f08e7b0d9f0997fa254fc5e52eda27d
- Verified independently with Windows `certutil` and Python `hashlib` (match: True)
-    - **Cleaned file:** data/appointments_clean.csv (110,516 × 15), SHA-256 ab148bd84e05fc1528cc2889f82f14b33ea379770a87f261e68292c4a5bfa0f5

## Verified contents (01_no_show_analysis.ipynb)
- 110,527 rows × 14 columns; 0 missing values; 0 duplicate rows
- 110,527 unique appointments; 62,299 unique patients (~1.8 appointments per patient)
- Appointment dates: 2016-04-29 to 2016-06-08 (~6 weeks)
- Booking dates: 2015-11-10 to 2016-06-08

## Discrepancies with the Kaggle description
1. **No-show rate:** Kaggle's headline says 30%; the data shows **20.19%**.
2. **Date columns:** Kaggle's data dictionary describes ScheduledDay and AppointmentDay
   the wrong way round. The data shows ScheduledDay = booking date/time,
   AppointmentDay = visit date (time always 00:00:00).
3. **Target coding:** No-show = "Yes" means the patient did **not** attend.

## Environment
- Dedicated conda environment `noshow` (conda-forge): Python 3.12.14, pandas 3.0.6,
  scikit-learn 1.9.1, matplotlib 3.11.2, seaborn 0.13.2
- Note: the default Anaconda environment (Python 3.14.6) failed to start a Jupyter
  kernel on Windows (ZeroMQ "Bad file descriptor"), so a dedicated 3.12 environment is used.