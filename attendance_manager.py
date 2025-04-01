import os
import csv
import datetime
import pandas as pd
from datetime import datetime, timedelta
import pytz
import config

class AttendanceManager:
    """Class to manage employee attendance records using Vietnamese time."""

    def __init__(self):
        """Initialize the attendance management system."""
        self.records_dir = config.ATTENDANCE_RECORDS_DIR
        self.work_start_time = config.WORK_START_TIME
        self.work_end_time = config.WORK_END_TIME
        self.late_threshold = config.LATE_THRESHOLD_MINUTES
        
        # Set Vietnam timezone (GMT+7)
        self.vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')
        
        # Ensure attendance records directory exists
        os.makedirs(self.records_dir, exist_ok=True)
        
        # Store today's date in Vietnam time for reference
        self.today = datetime.now(self.vietnam_tz).date()
        self.today_str = self.today.strftime('%Y-%m-%d')
        
        # Initialize or load today's attendance file
        self.today_file = os.path.join(self.records_dir, f'attendance_{self.today_str}.csv')
        self._initialize_daily_file()
        
        # Cache for employees who have already checked in today
        self.checked_in_employees = self._get_checked_in_employees()
    
    def _initialize_daily_file(self):
        """Initialize the daily attendance file if it doesn't exist."""
        if not os.path.exists(self.today_file):
            with open(self.today_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Employee ID', 'Name', 'Check-in Time', 'Check-out Time', 'Status', 'Comments'])
    
    def _get_checked_in_employees(self):
        """Get a dictionary of employees who have already checked in today."""
        if not os.path.exists(self.today_file):
            return {}
            
        checked_in = {}
        df = pd.read_csv(self.today_file)
        
        for _, row in df.iterrows():
            employee_id = row['Employee ID']
            # Only include employees who have checked in but not checked out
            if pd.notna(row['Check-in Time']) and pd.isna(row['Check-out Time']):
                checked_in[employee_id] = {
                    'name': row['Name'],
                    'check_in_time': row['Check-in Time']
                }
        
        return checked_in
    
    def _parse_time(self, time_str):
        """Parse time string to datetime.time object."""
        return datetime.strptime(time_str, '%H:%M').time()
    
    def _get_status(self, check_in_time):
        """Determine attendance status based on check-in time."""
        check_in_dt = datetime.strptime(check_in_time, '%H:%M:%S')
        work_start_dt = datetime.strptime(self.work_start_time, '%H:%M')
        
        # Calculate time difference in minutes
        time_diff = (check_in_dt - work_start_dt).total_seconds() / 60
        
        if time_diff <= 0:  # On time or early
            return 'On Time'
        elif time_diff <= self.late_threshold:  # Within grace period
            return 'Grace Period'
        else:  # Late
            return 'Late'
    
    def record_check_in(self, employee_id, name):
        """Record employee check-in.

        Args:
            employee_id (str): Unique identifier for the employee
            name (str): Name of the employee

        Returns:
            dict: Check-in details including status and time
        """
        current_time = datetime.now()
        time_str = current_time.strftime('%H:%M:%S')
        
        # Check if employee already checked in today
        if employee_id in self.checked_in_employees:
            return {
                'success': False,
                'message': f'{name} has already checked in today at {self.checked_in_employees[employee_id]["check_in_time"]}.'
            }
        
        # Determine attendance status
        status = self._get_status(time_str)
        
        # Prepare comments based on status
        comments = ''
        if status == 'Late':
            comments = f'Late by more than {self.late_threshold} minutes'
        elif status == 'Grace Period':
            comments = f'Within {self.late_threshold} minute grace period'
        
        # Append to CSV file
        with open(self.today_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([employee_id, name, time_str, '', status, comments])
        
        # Update cache
        self.checked_in_employees[employee_id] = {
            'name': name,
            'check_in_time': time_str
        }
        
        return {
            'success': True,
            'employee_id': employee_id,
            'name': name,
            'time': time_str,
            'status': status,
            'comments': comments
        }
    
    def record_check_out(self, employee_id, name):
        """Record employee check-out.

        Args:
            employee_id (str): Unique identifier for the employee
            name (str): Name of the employee

        Returns:
            dict: Check-out details including time and duration
        """
        current_time = datetime.now()
        time_str = current_time.strftime('%H:%M:%S')
        
        # Check if employee has checked in today
        if employee_id not in self.checked_in_employees:
            return {
                'success': False,
                'message': f'{name} has not checked in today.'
            }
        
        # Load the CSV file
        df = pd.read_csv(self.today_file)
        
        # Find the employee's row where check-out is empty
        mask = (df['Employee ID'] == employee_id) & (df['Check-out Time'].isna())
        if not mask.any():
            return {
                'success': False,
                'message': f'No active check-in found for {name}.'
            }
        
        # Update check-out time
        df.loc[mask, 'Check-out Time'] = time_str
        
        # Calculate work duration
        check_in_time = df.loc[mask, 'Check-in Time'].iloc[0]
        check_in_dt = datetime.strptime(check_in_time, '%H:%M:%S')
        check_out_dt = datetime.strptime(time_str, '%H:%M:%S')
        
        # Handle case where check-out is next day (after midnight)
        if check_out_dt < check_in_dt:
            check_out_dt += timedelta(days=1)
        
        duration = check_out_dt - check_in_dt
        duration_str = str(duration)
        
        # Add duration to comments
        current_comments = df.loc[mask, 'Comments'].iloc[0]
        if pd.isna(current_comments) or current_comments == '':
            df.loc[mask, 'Comments'] = f'Work duration: {duration_str}'
        else:
            df.loc[mask, 'Comments'] = f'{current_comments}; Work duration: {duration_str}'
        
        # Save the updated CSV
        df.to_csv(self.today_file, index=False)
        
        # Remove from checked-in cache
        self.checked_in_employees.pop(employee_id, None)
        
        return {
            'success': True,
            'employee_id': employee_id,
            'name': name,
            'time': time_str,
            'duration': duration_str
        }
    
    def get_attendance_summary(self, date=None):
        """Get attendance summary for a specific date.

        Args:
            date (str, optional): Date in 'YYYY-MM-DD' format. Defaults to today.

        Returns:
            pd.DataFrame: Attendance summary
        """
        if date is None:
            date = self.today_str
        
        file_path = os.path.join(self.records_dir, f'attendance_{date}.csv')
        
        if not os.path.exists(file_path):
            return pd.DataFrame(columns=['Employee ID', 'Name', 'Check-in Time', 'Check-out Time', 'Status', 'Comments'])
        
        return pd.read_csv(file_path)
    
    def generate_monthly_report(self, year=None, month=None):
        """Generate a monthly attendance report.

        Args:
            year (int, optional): Year for the report. Defaults to current year.
            month (int, optional): Month for the report. Defaults to current month.

        Returns:
            str: Path to the generated report file
        """
        # Default to current year and month if not specified
        if year is None:
            year = self.today.year
        if month is None:
            month = self.today.month
        
        # Create a date range for the month
        if month == 12:
            next_year = year + 1
            next_month = 1
        else:
            next_year = year
            next_month = month + 1
        
        start_date = datetime(year, month, 1).date()
        end_date = datetime(next_year, next_month, 1).date() - timedelta(days=1)
        
        # Initialize data structure for the report
        all_employees = {}
        dates_in_month = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') 
                         for i in range((end_date - start_date).days + 1)]
        
        # Collect data for each day in the month
        for date_str in dates_in_month:
            file_path = os.path.join(self.records_dir, f'attendance_{date_str}.csv')
            
            if not os.path.exists(file_path):
                continue
            
            df = pd.read_csv(file_path)
            
            for _, row in df.iterrows():
                employee_id = row['Employee ID']
                name = row['Name']
                status = row['Status']
                
                if employee_id not in all_employees:
                    all_employees[employee_id] = {'Name': name}
                    # Initialize all dates as 'Absent'
                    for d in dates_in_month:
                        all_employees[employee_id][d] = 'Absent'
                
                # Update the status for this date
                all_employees[employee_id][date_str] = status
        
        # Create DataFrame from the collected data
        if not all_employees:  # If no data found
            columns = ['Employee ID', 'Name'] + dates_in_month
            monthly_df = pd.DataFrame(columns=columns)
        else:
            rows = []
            for employee_id, data in all_employees.items():
                row = {'Employee ID': employee_id, 'Name': data['Name']}
                for date_str in dates_in_month:
                    row[date_str] = data.get(date_str, 'Absent')
                rows.append(row)
            
            monthly_df = pd.DataFrame(rows)
        
        # Save the report
        report_filename = f'monthly_report_{year}_{month:02d}.csv'
        report_path = os.path.join(self.records_dir, report_filename)
        monthly_df.to_csv(report_path, index=False)
        
        return report_path
