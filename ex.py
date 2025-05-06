import tkinter as tk
from tkinter import ttk, messagebox, PhotoImage, Canvas, filedialog, StringVar, BooleanVar
import tkinter.font as tkFont
import threading
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import json
import datetime
import time
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
import webbrowser
import sys
import random
import platform

# Check if required packages are installed
try:
    from ultralytics import YOLO
    import mediapipe as mp
except ImportError:
    print("Required packages not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "ultralytics", "mediapipe", "pillow", "matplotlib"])
    from ultralytics import YOLO
    import mediapipe as mp

class ExercisePoseApp:
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.load_resources()
        self.initialize_variables()
        self.create_styles()
        self.create_main_frame()
        self.create_dashboard()
        self.load_user_data()
        
    def setup_window(self):
        self.root.title("Advanced Exercise Posture Assistant")
        self.root.configure(bg="#121212")
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        app_width = min(1280, int(self.screen_width * 0.9))
        app_height = min(800, int(self.screen_height * 0.9))
        self.root.geometry(f"{app_width}x{app_height}")
        self.root.minsize(960, 600)
        
        # Make app responsive
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Set app icon if OS is not macOS (which handles icons differently)
        if platform.system() != "Darwin":
            try:
                self.root.iconbitmap("app_icon.ico")
            except:
                pass
    
    def load_resources(self):
        # Prepare data directory
        self.app_dir = os.path.join(os.path.expanduser("~"), "ExercisePoseApp")
        os.makedirs(self.app_dir, exist_ok=True)
        os.makedirs(os.path.join(self.app_dir, "data"), exist_ok=True)
        os.makedirs(os.path.join(self.app_dir, "profiles"), exist_ok=True)
        
        # Load YOLO model
        try:
            model_path = os.path.join(self.app_dir, "best.pt")
            if os.path.exists(model_path):
                self.model = YOLO(model_path).to('cpu')
            else:
                # Use default model
                self.model = YOLO("best.pt").to('cpu')
        except Exception as e:
            self.model = None
            print(f"Error loading model: {e}")
        
        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        
        # Load exercise data
        self.exercise_data = {
            "bicep curl": {
                "name": "Bicep Curl",
                "description": "Trains biceps by flexing at the elbow",
                "keypoints": [11, 13, 15],  # shoulder, elbow, wrist
                "target_angles": {"min": 30, "max": 160, "ideal": 90},
                "level": "beginner",
                "muscles": ["Biceps", "Forearms"],
                "color": "#2980b9",
                "feedback": {
                    "too_extended": "Bend your arm more",
                    "too_flexed": "Extend your arm more",
                    "good": "Good form! Keep going",
                    "complete": "Curl complete. Great job!"
                }
            },
            "push-up": {
                "name": "Push-Up",
                "description": "Full body exercise focusing on chest and arms",
                "keypoints": [11, 13, 15, 23, 25, 27],  # various upper body points
                "target_angles": {"min": 80, "max": 160, "ideal": 90},
                "level": "intermediate",
                "muscles": ["Chest", "Shoulders", "Triceps", "Core"],
                "color": "#27ae60",
                "feedback": {
                    "too_high": "Lower your body more",
                    "too_low": "Push up more",
                    "back_bent": "Keep your back straight",
                    "good": "Good form! Keep your core tight"
                }
            },
            "squat": {
                "name": "Squat",
                "description": "Lower body exercise for legs and glutes",
                "keypoints": [23, 25, 27],  # hip, knee, ankle
                "target_angles": {"min": 70, "max": 170, "ideal": 110},
                "level": "intermediate",
                "muscles": ["Quadriceps", "Hamstrings", "Glutes", "Core"],
                "color": "#f39c12",
                "feedback": {
                    "too_high": "Squat deeper",
                    "too_low": "Rise up more",
                    "knees_forward": "Keep knees behind toes",
                    "good": "Great depth! Keep chest up"
                }
            },
            "shoulder press": {
                "name": "Shoulder Press",
                "description": "Upper body exercise targeting shoulders",
                "keypoints": [11, 13, 15],  # shoulder, elbow, wrist
                "target_angles": {"min": 60, "max": 170, "ideal": 160},
                "level": "advanced",
                "muscles": ["Shoulders", "Triceps", "Upper Back"],
                "color": "#8e44ad",
                "feedback": {
                    "too_low": "Push higher",
                    "elbows_out": "Keep elbows forward",
                    "good": "Good form! Keep pushing",
                    "locked_out": "Great lockout position"
                }
            }
        }
        
    def initialize_variables(self):
        self.camera_active = False
        self.camera_id = 0
        self.cap = None
        self.current_exercise = None
        self.angle_buffer = deque(maxlen=10)
        self.prev_hip_y = None
        self.session_start_time = None
        self.session_reps = 0
        self.session_angles = []
        self.session_feedback = []
        self.use_voice_feedback = BooleanVar(value=False)
        self.show_skeleton = BooleanVar(value=True)
        self.show_angles = BooleanVar(value=True)
        self.camera_frame = None
        self.processed_frame = None
        self.canvas_width = 640
        self.canvas_height = 480
        self.current_user = "Default User"
        self.dark_mode = BooleanVar(value=True)
        self.calibration_mode = False
        self.calibration_points = []
        
        # Create a dictionary to store theme colors
        self.update_theme_colors()
        
    def update_theme_colors(self):
        if self.dark_mode.get():
            self.theme = {
                "bg_main": "#121212",
                "bg_secondary": "#1e1e1e",
                "bg_tertiary": "#2d2d2d",
                "text_primary": "#ffffff",
                "text_secondary": "#bbbbbb",
                "accent": "#8e44ad",
                "highlight": "#f39c12",
                "success": "#27ae60",
                "warning": "#e67e22",
                "error": "#e74c3c",
                "shadow": "black"
            }
        else:
            self.theme = {
                "bg_main": "#f5f5f5",
                "bg_secondary": "#ffffff",
                "bg_tertiary": "#e0e0e0",
                "text_primary": "#212121",
                "text_secondary": "#757575",
                "accent": "#9c27b0",
                "highlight": "#f39c12",
                "success": "#2ecc71",
                "warning": "#f39c12",
                "error": "#e74c3c",
                "shadow": "#bbbbbb"
            }
        
    def create_styles(self):
        # Create custom fonts
        self.title_font = tkFont.Font(family="Helvetica", size=24, weight="bold")
        self.heading_font = tkFont.Font(family="Helvetica", size=18, weight="bold")
        self.subheading_font = tkFont.Font(family="Helvetica", size=14, weight="bold")
        self.normal_font = tkFont.Font(family="Helvetica", size=12)
        self.small_font = tkFont.Font(family="Helvetica", size=10)
        
        # Configure ttk styles
        self.style = ttk.Style()
        self.style.configure("TFrame", background=self.theme["bg_main"])
        self.style.configure("TLabel", background=self.theme["bg_main"], foreground=self.theme["text_primary"])
        self.style.configure("TButton", font=self.normal_font, padding=10)
        
        # Custom button styles
        self.style.configure("Primary.TButton", background=self.theme["accent"], foreground="white")
        self.style.configure("Secondary.TButton", background=self.theme["bg_tertiary"], foreground=self.theme["text_primary"])
        self.style.configure("Success.TButton", background=self.theme["success"], foreground="white")
        self.style.configure("Warning.TButton", background=self.theme["warning"], foreground="white")
        self.style.configure("Danger.TButton", background=self.theme["error"], foreground="white")
        
        # Configure progress bar style
        self.style.configure("TProgressbar", thickness=15, background=self.theme["accent"])
    
    def create_main_frame(self):
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(0, weight=1)
        
        # Create navigation frame
        self.nav_frame = tk.Frame(self.main_frame, bg=self.theme["bg_secondary"], width=200)
        self.nav_frame.grid(row=0, column=0, sticky="nsw", padx=0, pady=0)
        
        # Create app title
        title_frame = tk.Frame(self.nav_frame, bg=self.theme["accent"], padx=10, pady=20)
        title_frame.pack(fill='x')
        
        title_label = tk.Label(title_frame, text="Exercise\nPosture Pro", font=self.title_font, 
                              bg=self.theme["accent"], fg="white")
        title_label.pack(pady=5)
        
        # Create navigation buttons
        self.create_nav_button("Dashboard", self.show_dashboard)
        self.create_nav_button("Start Exercise", self.show_exercise_selection)
        self.create_nav_button("Progress Tracker", self.show_progress)
        self.create_nav_button("Settings", self.show_settings)
        self.create_nav_button("Help", self.show_help)
        
        # Add app version and theme toggle at bottom
        version_frame = tk.Frame(self.nav_frame, bg=self.theme["bg_secondary"], padx=10, pady=10)
        version_frame.pack(side="bottom", fill='x')
        
        version_label = tk.Label(version_frame, text="Version 2.0", font=self.small_font, 
                                bg=self.theme["bg_secondary"], fg=self.theme["text_secondary"])
        version_label.pack(side="left", pady=5)
        
        theme_toggle = ttk.Checkbutton(version_frame, text="", variable=self.dark_mode, 
                                    command=self.toggle_theme, style="Switch.TCheckbutton")
        theme_toggle.pack(side="right", pady=5)
        
        # Create content frame (where different pages will be shown)
        self.content_frame = tk.Frame(self.main_frame, bg=self.theme["bg_main"])
        self.content_frame.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        self.main_frame.columnconfigure(1, weight=4)  # Content takes most space
        
    def create_nav_button(self, text, command):
        button_frame = tk.Frame(self.nav_frame, bg=self.theme["bg_secondary"])
        button_frame.pack(fill='x')
        
        button = tk.Button(button_frame, text=text, font=self.normal_font,
                         bg=self.theme["bg_secondary"], fg=self.theme["text_primary"],
                         activebackground=self.theme["accent"], activeforeground="white",
                         bd=0, padx=20, pady=15, anchor="w", width=20,
                         command=command)
        button.pack(fill='x')
    
    def toggle_theme(self):
        self.update_theme_colors()
        self.clear_frame(self.content_frame)
        if self.current_page == "dashboard":
            self.create_dashboard()
        elif self.current_page == "exercise_selection":
            self.show_exercise_selection()
        elif self.current_page == "progress":
            self.show_progress()
        elif self.current_page == "settings":
            self.show_settings()
        elif self.current_page == "help":
            self.show_help()
        elif self.current_page == "exercise_view":
            self.show_exercise_view(self.current_exercise)
    
    def create_dashboard(self):
        self.current_page = "dashboard"
        self.clear_frame(self.content_frame)
        
        # Create welcome header
        welcome_frame = tk.Frame(self.content_frame, bg=self.theme["bg_main"], padx=20, pady=20)
        welcome_frame.pack(fill='x')
        
        welcome_label = tk.Label(welcome_frame, text=f"Welcome, {self.current_user}",
                                font=self.heading_font, bg=self.theme["bg_main"], fg=self.theme["text_primary"])
        welcome_label.pack(anchor="w")
        
        date_label = tk.Label(welcome_frame, text=datetime.datetime.now().strftime("%A, %B %d, %Y"),
                            font=self.normal_font, bg=self.theme["bg_main"], fg=self.theme["text_secondary"])
        date_label.pack(anchor="w")
        
        # Create quick start section
        quick_start_frame = tk.Frame(self.content_frame, bg=self.theme["bg_main"], padx=20, pady=10)
        quick_start_frame.pack(fill='x')
        
        quick_start_label = tk.Label(quick_start_frame, text="Quick Start Exercise",
                                   font=self.subheading_font, bg=self.theme["bg_main"], fg=self.theme["text_primary"])
        quick_start_label.pack(anchor="w", pady=(0, 10))
        
        # Create grid for exercise cards
        exercise_grid = tk.Frame(quick_start_frame, bg=self.theme["bg_main"])
        exercise_grid.pack(fill='x')
        
        # Create exercise cards
        col = 0
        for exercise_key, exercise_info in self.exercise_data.items():
            self.create_exercise_card(exercise_grid, exercise_key, col)
            col += 1
        
        # Create recent activity section
        recent_frame = tk.Frame(self.content_frame, bg=self.theme["bg_main"], padx=20, pady=10)
        recent_frame.pack(fill='x', pady=20)
        
        recent_label = tk.Label(recent_frame, text="Recent Activity",
                             font=self.subheading_font, bg=self.theme["bg_main"], fg=self.theme["text_primary"])
        recent_label.pack(anchor="w", pady=(0, 10))
        
        # Create empty state or load recent activities
        recent_activities = self.load_recent_activities()
        if recent_activities:
            for activity in recent_activities[:3]:  # Show only last 3 activities
                self.create_activity_item(recent_frame, activity)
        else:
            no_activity = tk.Label(recent_frame, text="No recent activities. Start exercising to track your progress!",
                                 font=self.normal_font, bg=self.theme["bg_tertiary"], fg=self.theme["text_secondary"],
                                 padx=15, pady=30)
            no_activity.pack(fill='x', padx=5, pady=5)
        
        # Create tips section
        tips_frame = tk.Frame(self.content_frame, bg=self.theme["bg_main"], padx=20, pady=10)
        tips_frame.pack(fill='x', pady=10)
        
        tips_label = tk.Label(tips_frame, text="Exercise Tips",
                           font=self.subheading_font, bg=self.theme["bg_main"], fg=self.theme["text_primary"])
        tips_label.pack(anchor="w", pady=(0, 10))
        
        tips = [
            "Remember to warm up before starting any exercise",
            "Stay hydrated during your workout sessions",
            "Maintain proper form for maximum effectiveness",
            "Track your progress to stay motivated"
        ]
        
        tip_box = tk.Frame(tips_frame, bg=self.theme["bg_tertiary"], padx=15, pady=15,
                         highlightbackground=self.theme["accent"], highlightthickness=1)
        tip_box.pack(fill='x')
        
        tip_text = tk.Label(tip_box, text=random.choice(tips), font=self.normal_font,
                          bg=self.theme["bg_tertiary"], fg=self.theme["text_primary"], wraplength=800)
        tip_text.pack(anchor="w")
    
    def create_exercise_card(self, parent, exercise_key, col):
        exercise = self.exercise_data[exercise_key]
        
        # Create card frame
        card = tk.Frame(parent, bg=self.theme["bg_secondary"], padx=15, pady=15,
                      highlightbackground=exercise["color"], highlightthickness=1)
        card.grid(row=0, column=col, padx=10, pady=10, sticky="nsew")
        
        # Add exercise name
        name_label = tk.Label(card, text=exercise["name"], font=self.subheading_font,
                            bg=self.theme["bg_secondary"], fg=self.theme["text_primary"])
        name_label.pack(anchor="w")
        
        # Add difficulty level
        level_label = tk.Label(card, text=exercise["level"].capitalize(),
                             font=self.small_font, bg=self.theme["bg_secondary"], fg=self.theme["text_secondary"])
        level_label.pack(anchor="w", pady=(0, 10))
        
        # Add muscles worked
        muscles_text = ", ".join(exercise["muscles"])
        muscles_label = tk.Label(card, text=f"Targets: {muscles_text}",
                               font=self.small_font, bg=self.theme["bg_secondary"], fg=self.theme["text_secondary"],
                               wraplength=180)
        muscles_label.pack(anchor="w", pady=(0, 10))
        
        # Add start button
        start_button = tk.Button(card, text="Start", font=self.normal_font,
                               bg=exercise["color"], fg="white", padx=10, pady=5,
                               command=lambda e=exercise_key: self.show_exercise_view(e))
        start_button.pack(anchor="w")
    
    def create_activity_item(self, parent, activity):
        # Create activity item frame
        item_frame = tk.Frame(parent, bg=self.theme["bg_tertiary"], padx=15, pady=15)
        item_frame.pack(fill='x', padx=5, pady=5)
        
        # Add activity date/time
        date_label = tk.Label(item_frame, text=activity["date"], font=self.small_font,
                            bg=self.theme["bg_tertiary"], fg=self.theme["text_secondary"])
        date_label.pack(anchor="w")
        
        # Add activity details
        detail_frame = tk.Frame(item_frame, bg=self.theme["bg_tertiary"])
        detail_frame.pack(fill='x', pady=(5, 0))
        
        exercise_label = tk.Label(detail_frame, text=activity["exercise"],
                                font=self.normal_font, bg=self.theme["bg_tertiary"], fg=self.theme["text_primary"])
        exercise_label.pack(side="left")
        
        duration_label = tk.Label(detail_frame, text=f"{activity['duration']} min",
                                font=self.normal_font, bg=self.theme["bg_tertiary"], fg=self.theme["text_secondary"])
        duration_label.pack(side="right")
        
        # Add performance bar if available
        if "performance" in activity:
            perf_frame = tk.Frame(item_frame, bg=self.theme["bg_tertiary"], pady=5)
            perf_frame.pack(fill='x')
            
            perf_text = tk.Label(perf_frame, text="Performance:",
                               font=self.small_font, bg=self.theme["bg_tertiary"], fg=self.theme["text_secondary"])
            perf_text.pack(side="left")
            
            perf_val = tk.Label(perf_frame, text=f"{activity['performance']}%",
                              font=self.small_font, bg=self.theme["bg_tertiary"], fg=self.theme["text_primary"])
            perf_val.pack(side="right")
            
            # Performance bar background
            bar_bg = tk.Frame(item_frame, bg=self.theme["bg_secondary"], height=8)
            bar_bg.pack(fill='x', pady=(0, 5))
            
            # Performance bar fill
            performance = min(100, max(0, activity["performance"]))
            bar_fill = tk.Frame(bar_bg, bg=self.get_performance_color(performance), height=8, width=int(bar_bg.winfo_reqwidth() * performance / 100))
            bar_fill.pack(side="left", anchor="w")
    
    def get_performance_color(self, performance):
        if performance >= 80:
            return self.theme["success"]
        elif performance >= 60:
            return self.theme["warning"]
        else:
            return self.theme["error"]
    
    def clear_frame(self, frame):
        # Destroy all widgets in the frame
        for widget in frame.winfo_children():
            widget.destroy()
    
    def show_dashboard(self):
        self.clear_frame(self.content_frame)
        self.create_dashboard()
    
    def show_exercise_selection(self):
        self.current_page = "exercise_selection"
        self.clear_frame(self.content_frame)
        
        # Create header
        header_frame = tk.Frame(self.content_frame, bg=self.theme["bg_main"], padx=20, pady=20)
        header_frame.pack(fill='x')
        
        header_label = tk.Label(header_frame, text="Select an Exercise",
                               font=self.heading_font, bg=self.theme["bg_main"], fg=self.theme["text_primary"])
        header_label.pack(anchor="w")
        
        # Create grid for exercise detail cards
        exercises_frame = tk.Frame(self.content_frame, bg=self.theme["bg_main"], padx=20, pady=10)
        exercises_frame.pack(fill='both', expand=True)
        
        row, col = 0, 0
        for i, (exercise_key, exercise_info) in enumerate(self.exercise_data.items()):
            if i % 2 == 0 and i > 0:
                row += 1
                col = 0
                
            self.create_exercise_detail_card(exercises_frame, exercise_key, exercise_info, row, col)
            col += 1
    
    def create_exercise_detail_card(self, parent, exercise_key, exercise_info, row, col):
        # Create card frame
        card = tk.Frame(parent, bg=self.theme["bg_secondary"], padx=20, pady=20,
                      highlightbackground=exercise_info["color"], highlightthickness=1)
        card.grid(row=row, column=col, padx=15, pady=15, sticky="nsew")
        
        # Add exercise name with colored indicator
        name_frame = tk.Frame(card, bg=self.theme["bg_secondary"])
        name_frame.pack(fill='x', pady=(0, 10))
        
        indicator = tk.Frame(name_frame, bg=exercise_info["color"], width=5, height=25)
        indicator.pack(side="left", padx=(0, 10))
        
        name_label = tk.Label(name_frame, text=exercise_info["name"], font=self.subheading_font,
                            bg=self.theme["bg_secondary"], fg=self.theme["text_primary"])
        name_label.pack(side="left")
        
        # Add difficulty level
        level_frame = tk.Frame(card, bg=self.theme["bg_secondary"])
        level_frame.pack(fill='x', pady=(0, 10))
        
        level_label = tk.Label(level_frame, text="Difficulty:",
                             font=self.small_font, bg=self.theme["bg_secondary"], fg=self.theme["text_secondary"])
        level_label.pack(side="left")
        
        # Add level indicator
        level_indicator = tk.Frame(level_frame, bg=self.theme["bg_secondary"])
        level_indicator.pack(side="left", padx=(5, 0))
        
        level = exercise_info["level"]
        colors = {"beginner": self.theme["success"], 
                 "intermediate": self.theme["warning"], 
                 "advanced": self.theme["error"]}
        levels = {"beginner": 1, "intermediate": 2, "advanced": 3}
        
        for i in range(3):
            indicator = tk.Frame(level_indicator, bg=colors[level] if i < levels[level] else self.theme["bg_tertiary"],
                               width=15, height=8)
            indicator.pack(side="left", padx=2)
        
        level_text = tk.Label(level_frame, text=level.capitalize(),
                            font=self.small_font, bg=self.theme["bg_secondary"], fg=colors[level])
        level_text.pack(side="left", padx=(5, 0))
        
        # Add description
        desc_label = tk.Label(card, text=exercise_info["description"],
                            font=self.normal_font, bg=self.theme["bg_secondary"], fg=self.theme["text_primary"],
                            wraplength=350, justify="left")
        desc_label.pack(anchor="w", pady=(0, 15))
        
        # Add muscles targeted
        muscles_frame = tk.Frame(card, bg=self.theme["bg_secondary"])
        muscles_frame.pack(fill='x', pady=(0, 15))
        
        muscles_label = tk.Label(muscles_frame, text="Muscles:",
                               font=self.small_font, bg=self.theme["bg_secondary"], fg=self.theme["text_secondary"])
        muscles_label.pack(side="left")
        
        muscles_text = ", ".join(exercise_info["muscles"])
        muscles_value = tk.Label(muscles_frame, text=muscles_text,
                               font=self.small_font, bg=self.theme["bg_secondary"], fg=self.theme["text_primary"])
        muscles_value.pack(side="left", padx=(5, 0))
        
        # Add target angle
        angle_frame = tk.Frame(card, bg=self.theme["bg_secondary"])
        angle_frame.pack(fill='x', pady=(0, 15))
        
        angle_label = tk.Label(angle_frame, text="Ideal Angle:",
                             font=self.small_font, bg=self.theme["bg_secondary"], fg=self.theme["text_secondary"])
        angle_label.pack(side="left")
        
        angle_value = tk.Label(angle_frame, text=f"{exercise_info['target_angles']['ideal']}°",
                             font=self.small_font, bg=self.theme["bg_secondary"], fg=self.theme["text_primary"])
        angle_value.pack(side="left", padx=(5, 0))
        
        # Add buttons
        button_frame = tk.Frame(card, bg=self.theme["bg_secondary"])
        button_frame.pack(fill='x')
        
        start_button = tk.Button(button_frame, text="Start Exercise", font=self.normal_font,
                               bg=exercise_info["color"], fg="white", padx=15, pady=8,
                               command=lambda e=exercise_key: self.show_exercise_view(e))
        start_button.pack(side="right")
        
        demo_button = tk.Button(button_frame, text="View Demo", font=self.normal_font,
                              bg=self.theme["bg_tertiary"], fg=self.theme["text_primary"], padx=15, pady=8,
                              command=lambda e=exercise_info["name"]: self.show_demo(e))
        demo_button.pack(side="right", padx=(0, 10))
    
    def show_demo(self, exercise_name):
        # In a real app, this would display a video or animation
        # For this example, just show a message
        messagebox.showinfo("Demo", f"This would show a demonstration video for {exercise_name}.")
    
    def show_exercise_view(self, exercise_key):
        self.current_page = "exercise_view"
        self.current_exercise = exercise_key
        self.clear_frame(self.content_frame)
        exercise = self.exercise_data[exercise_key]
        
        # Create header
        header_frame = tk.Frame(self.content_frame, bg=self.theme["bg_main"], padx=20, pady=20)
        header_frame.pack(fill='x')
        
        # Add back button
        back_button = tk.Button(header_frame, text="← Back", font=self.normal_font,
                              bg=self.theme["bg_tertiary"], fg=self.theme["text_primary"], padx=10, pady=5,
                              command=self.show_dashboard)
        back_button.pack(side="left")
        
        # Add exercise name
        title_label = tk.Label(header_frame, text=exercise["name"], font=self.heading_font,
                             bg=self.theme["bg_main"], fg=self.theme["text_primary"])
        title_label.pack(side="left", padx=(20, 0))
        
        # Create main content with camera feed and controls
        content_frame = tk.Frame(self.content_frame, bg=self.theme["bg_main"], padx=20)
        content_frame.pack(fill='both', expand=True)
        
        # Left panel for camera feed
        left_panel = tk.Frame(content_frame, bg=self.theme["bg_secondary"])
        left_panel.pack(side="left", fill='both', expand=True, padx=(0, 10))
        
        # Camera canvas
        camera_frame = tk.Frame(left_panel, bg="black")
        camera_frame.pack(pady=10, padx=10)
        
        self.camera_canvas = tk.Canvas(camera_frame, width=self.canvas_width, height=self.canvas_height, bg="black",
                                     highlightthickness=0)
        self.camera_canvas.pack()
        
        # Placeholder for camera
        placeholder_text = tk.Label(self.camera_canvas, text="Camera feed will appear here",
                                  font=self.normal_font, bg="black", fg="white")
        placeholder_text.place(relx=0.5, rely=0.5, anchor="center")
        
        # Camera controls
        controls_frame = tk.Frame(left_panel, bg=self.theme["bg_secondary"], padx=10, pady=10)
        controls_frame.pack(fill='x')
        
        # Start/Stop camera button
        self.camera_button = tk.Button(controls_frame, text="Start Camera", font=self.normal_font,
                                     bg=self.theme["success"], fg="white", padx=15, pady=5,
                                     command=self.toggle_camera)
        self.camera_button.pack(side="left")
        
        # Camera options
        camera_options = tk.Frame(controls_frame, bg=self.theme["bg_secondary"])
        camera_options.pack(side="right")
        
        # Show skeleton checkbox
        skeleton_check = ttk.Checkbutton(camera_options, text="Show Skeleton", variable=self.show_skeleton,
                                       style="TCheckbutton")
        skeleton_check.pack(side="left", padx=10)
        
        # Show angles checkbox
        angles_check = ttk.Checkbutton(camera_options, text="Show Angles", variable=self.show_angles,
                                     style="TCheckbutton")
        angles_check.pack(side="left", padx=10)
        
        # Right panel for stats and feedback
        right_panel = tk.Frame(content_frame, bg=self.theme["bg_tertiary"], width=300)
        right_panel.pack(side="right", fill='both', padx=(10, 0))
        
        # Add exercise details
        details_frame = tk.Frame(right_panel, bg=self.theme["bg_tertiary"], padx=15, pady=15)
        details_frame.pack(fill='x')
        
        details_label = tk.Label(details_frame, text="Exercise Details",
                               font=self.subheading_font, bg=self.theme["bg_tertiary"], fg=self.theme["text_primary"])
        details_label.pack(anchor="w", pady=(0, 10))
        
        # Muscles worked
        muscles_frame = tk.Frame(details_frame, bg=self.theme["bg_tertiary"])
        muscles_frame.pack(fill='x', pady=5)
        
        muscles_label = tk.Label(muscles_frame, text="Muscles:",
                               font=self.small_font, bg=self.theme["bg_tertiary"], fg=self.theme["text_secondary"])
        muscles_label.pack(side="left")
        
        muscles_text = ", ".join(exercise["muscles"])
        muscles_value = tk.Label(muscles_frame, text=muscles_text,
                               font=self.small_font, bg=self.theme["bg_tertiary"], fg=self.theme["text_primary"])
        muscles_value.pack(side="left", padx=(5, 0))
        
        # Target angle
        angle_frame = tk.Frame(details_frame, bg=self.theme["bg_tertiary"])
        angle_frame.pack(fill='x', pady=5)
        
        angle_label = tk.Label(angle_frame, text="Target Angle:",
                             font=self.small_font, bg=self.theme["bg_tertiary"], fg=self.theme["text_secondary"])
        angle_label.pack(side="left")
        
        angle_range = f"{exercise['target_angles']['min']}° - {exercise['target_angles']['max']}°"
        angle_value = tk.Label(angle_frame, text=angle_range,
                             font=self.small_font, bg=self.theme["bg_tertiary"], fg=self.theme["text_primary"])
        angle_value.pack(side="left", padx=(5, 0))
        
        # Difficulty
        diff_frame = tk.Frame(details_frame, bg=self.theme["bg_tertiary"])
        diff_frame.pack(fill='x', pady=5)
        
        diff_label = tk.Label(diff_frame, text="Difficulty:",
                            font=self.small_font, bg=self.theme["bg_tertiary"], fg=self.theme["text_secondary"])
        diff_label.pack(side="left")
        
        diff_value = tk.Label(diff_frame, text=exercise["level"].capitalize(),
                            font=self.small_font, bg=self.theme["bg_tertiary"], fg=self.theme["text_primary"])
        diff_value.pack(side="left", padx=(5, 0))
        
        # Add current stats
        stats_frame = tk.Frame(right_panel, bg=self.theme["bg_tertiary"], padx=15, pady=15)
        stats_frame.pack(fill='x', pady=10)
        
        stats_label = tk.Label(stats_frame, text="Current Session",
                             font=self.subheading_font, bg=self.theme["bg_tertiary"], fg=self.theme["text_primary"])
        stats_label.pack(anchor="w", pady=(0, 10))
        
        # Time counter
        time_frame = tk.Frame(stats_frame, bg=self.theme["bg_tertiary"])
        time_frame.pack(fill='x', pady=5)
        
        time_label = tk.Label(time_frame, text="Duration:",
                            font=self.small_font, bg=self.theme["bg_tertiary"], fg=self.theme["text_secondary"])
        time_label.pack(side="left")
        
        self.time_value = tk.Label(time_frame, text="00:00",
                                 font=self.small_font, bg=self.theme["bg_tertiary"], fg=self.theme["text_primary"])
        self.time_value.pack(side="left", padx=(5, 0))
        
        # Rep counter
        rep_frame = tk.Frame(stats_frame, bg=self.theme["bg_tertiary"])
        rep_frame.pack(fill='x', pady=5)
        
        rep_label = tk.Label(rep_frame, text="Repetitions:",
                           font=self.small_font, bg=self.theme["bg_tertiary"], fg=self.theme["text_secondary"])
        rep_label.pack(side="left")
        
        self.rep_value = tk.Label(rep_frame, text="0",
                                font=self.small_font, bg=self.theme["bg_tertiary"], fg=self.theme["text_primary"])
        self.rep_value.pack(side="left", padx=(5, 0))
        
        # Current angle
        curr_angle_frame = tk.Frame(stats_frame, bg=self.theme["bg_tertiary"])
        curr_angle_frame.pack(fill='x', pady=5)
        
        curr_angle_label = tk.Label(curr_angle_frame, text="Current Angle:",
                                  font=self.small_font, bg=self.theme["bg_tertiary"], fg=self.theme["text_secondary"])
        curr_angle_label.pack(side="left")
        
        self.curr_angle_value = tk.Label(curr_angle_frame, text="0°",
                                       font=self.small_font, bg=self.theme["bg_tertiary"], fg=self.theme["text_primary"])
        self.curr_angle_value.pack(side="left", padx=(5, 0))
        
        # Add feedback section
        feedback_frame = tk.Frame(right_panel, bg=self.theme["bg_tertiary"], padx=15, pady=15)
        feedback_frame.pack(fill='x', pady=10)
        
        feedback_label = tk.Label(feedback_frame, text="Feedback",
                                font=self.subheading_font, bg=self.theme["bg_tertiary"], fg=self.theme["text_primary"])
        feedback_label.pack(anchor="w", pady=(0, 10))
        
        self.feedback_box = tk.Label(feedback_frame, text="Start camera to begin receiving feedback",
                                   font=self.normal_font, bg=self.theme["bg_main"], fg=self.theme["text_primary"],
                                   wraplength=250, justify="center", padx=20, pady=20, height=5)
        self.feedback_box.pack(fill='x')
        
        # Add voice feedback option
        voice_frame = tk.Frame(right_panel, bg=self.theme["bg_tertiary"], padx=15, pady=15)
        voice_frame.pack(fill='x')
        
        voice_check = ttk.Checkbutton(voice_frame, text="Voice Feedback", variable=self.use_voice_feedback,
                                     style="TCheckbutton")
        voice_check.pack(side="left")
        
        # Add end session button
        end_button = tk.Button(right_panel, text="End Session", font=self.normal_font,
                             bg=self.theme["error"], fg="white", padx=15, pady=10,
                             command=self.end_exercise_session)
        end_button.pack(pady=20)
    
    def toggle_camera(self):
        if self.camera_active:
            self.stop_camera()
        else:
            self.start_camera()
    
    def start_camera(self):
        if self.cap is not None:
            return
            
        self.session_start_time = time.time()
        self.session_reps = 0
        self.angle_buffer.clear()
        self.session_angles = []
        self.session_feedback = []
        
        # Start camera
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open camera. Please check your camera settings.")
                self.cap = None
                return
                
            self.camera_active = True
            self.camera_button.config(text="Stop Camera", bg=self.theme["error"])
            
            # Start camera thread
            self.camera_thread = threading.Thread(target=self.process_camera_feed)
            self.camera_thread.daemon = True
            self.camera_thread.start()
            
            # Start timer update
            self.timer_thread = threading.Thread(target=self.update_session_time)
            self.timer_thread.daemon = True
            self.timer_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error starting camera: {str(e)}")
    
    def stop_camera(self):
        self.camera_active = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.camera_button.config(text="Start Camera", bg=self.theme["success"])
    
    def process_camera_feed(self):
        prev_angle = None
        direction = None
        rep_counted = False
        
        while self.camera_active and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Resize frame for display
            display_frame = cv2.resize(frame, (self.canvas_width, self.canvas_height))
            
            # Process with MediaPipe
            image_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            pose_results = self.pose.process(image_rgb)
            
            # Create a copy for drawing
            output_frame = display_frame.copy()
            
            if pose_results.pose_landmarks:
                # Extract landmarks
                landmarks = []
                for i, lm in enumerate(pose_results.pose_landmarks.landmark):
                    h, w, _ = output_frame.shape
                    px, py = int(lm.x * w), int(lm.y * h)
                    landmarks.append((px, py))
                    
                # Draw skeleton if enabled
                if self.show_skeleton.get():
                    self.mp_drawing.draw_landmarks(
                        output_frame, 
                        pose_results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        self.drawing_spec,
                        self.drawing_spec
                    )
                    
                # Calculate angle for current exercise
                exercise = self.exercise_data[self.current_exercise]
                
                # Get keypoints for current exercise
                keypoints = exercise["keypoints"]
                target_angles = exercise["target_angles"]
                
                # Calculate angle
                if len(keypoints) >= 3:
                    a = landmarks[keypoints[0]]
                    b = landmarks[keypoints[1]]
                    c = landmarks[keypoints[2]]
                    
                    angle = self.calculate_angle(a, b, c)
                    self.angle_buffer.append(angle)
                    avg_angle = sum(self.angle_buffer) / len(self.angle_buffer)
                    
                    # Store for analysis
                    self.session_angles.append(avg_angle)
                    
                    # Draw angle if enabled
                    if self.show_angles.get():
                        # Draw angle lines
                        cv2.line(output_frame, b, a, (0, 255, 0), 2)
                        cv2.line(output_frame, b, c, (0, 255, 0), 2)
                        
                        # Draw angle value
                        angle_text = f"{int(avg_angle)}°"
                        cv2.putText(output_frame, angle_text, 
                                   (b[0] - 50, b[1] + 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # Update angle display
                    self.root.after(10, lambda a=int(avg_angle): self.curr_angle_value.config(text=f"{a}°"))
                    
                    # Track rep counting
                    if prev_angle is not None:
                        if avg_angle > prev_angle:
                            current_direction = "up"
                        else:
                            current_direction = "down"
                            
                        # Detect rep for different exercises
                        if self.current_exercise == "bicep curl":
                            # Rep is complete when going from flexed to extended position
                            if (direction == "down" and current_direction == "up" and 
                                prev_angle < target_angles["min"] + 10):
                                if not rep_counted:
                                    self.session_reps += 1
                                    self.root.after(10, lambda r=self.session_reps: self.rep_value.config(text=str(r)))
                                    rep_counted = True
                            elif avg_angle > target_angles["max"] - 20:
                                rep_counted = False
                                
                        elif self.current_exercise == "push-up":
                            # Rep is complete when going from down to up position
                            if (direction == "up" and current_direction == "down" and 
                                prev_angle > target_angles["max"] - 20):
                                if not rep_counted:
                                    self.session_reps += 1
                                    self.root.after(10, lambda r=self.session_reps: self.rep_value.config(text=str(r)))
                                    rep_counted = True
                            elif avg_angle < target_angles["min"] + 20:
                                rep_counted = False
                                
                        elif self.current_exercise == "squat":
                            # Rep is complete when going from down to up position
                            if (direction == "up" and current_direction == "down" and 
                                prev_angle > target_angles["max"] - 20):
                                if not rep_counted:
                                    self.session_reps += 1
                                    self.root.after(10, lambda r=self.session_reps: self.rep_value.config(text=str(r)))
                                    rep_counted = True
                            elif avg_angle < target_angles["min"] + 20:
                                rep_counted = False
                                
                        elif self.current_exercise == "shoulder press":
                            # Rep is complete when going from down to up position
                            if (direction == "down" and current_direction == "up" and 
                                prev_angle > target_angles["max"] - 10):
                                if not rep_counted:
                                    self.session_reps += 1
                                    self.root.after(10, lambda r=self.session_reps: self.rep_value.config(text=str(r)))
                                    rep_counted = True
                            elif avg_angle < target_angles["min"] + 10:
                                rep_counted = False
                        
                        direction = current_direction
                    else:
                        direction = "none"
                    
                    prev_angle = avg_angle
                    
                    # Generate feedback
                    feedback = self.generate_feedback(avg_angle, target_angles, self.current_exercise)
                    self.session_feedback.append(feedback)
                    
                    # Show feedback
                    feedback_color = self.theme["success"] if "Good" in feedback else (
                                    self.theme["warning"] if "Complete" in feedback else self.theme["error"])
                    self.root.after(10, lambda f=feedback, c=feedback_color: 
                                   self.feedback_box.config(text=f, bg=c, fg="white"))
                
            # Convert frame for tkinter display
            self.processed_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(self.processed_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Update canvas with new frame
            self.root.after(10, lambda i=imgtk: self.update_camera_canvas(i))
            
            # Delay to reduce CPU usage
            time.sleep(0.03)
    
    def update_camera_canvas(self, imgtk):
        if self.camera_active:
            self.camera_canvas.create_image(0, 0, anchor="nw", image=imgtk)
            self.camera_canvas.image = imgtk
    
    def generate_feedback(self, angle, target_angles, exercise_type):
        feedback = "No feedback available"
        
        if exercise_type == "bicep curl":
            if angle > target_angles["max"] - 10:
                feedback = "Extend more"
            elif angle < target_angles["min"] + 10:
                feedback = "Curl Complete"
            else:
                feedback = "Good Form"
                
        elif exercise_type == "push-up":
            if angle > target_angles["max"] - 10:
                feedback = "Keep body straight"
            elif angle < target_angles["min"] + 10:
                feedback = "Good Form"
            else:
                feedback = "Adjust your posture"
                
        elif exercise_type == "squat":
            if angle < target_angles["min"] + 10:
                feedback = "Too Low"
            elif angle > target_angles["max"] - 10:
                feedback = "Stand Up Straight"
            else:
                feedback = "Good Squat"
                
        elif exercise_type == "shoulder press":
            if angle > target_angles["max"] - 10:
                feedback = "Locked Out"
            elif angle < target_angles["min"] + 10:
                feedback = "Push Higher"
            else:
                feedback = "Good Form"
                
        return feedback
    
    def update_session_time(self):
        while self.camera_active:
            if self.session_start_time:
                elapsed = int(time.time() - self.session_start_time)
                minutes = elapsed // 60
                seconds = elapsed % 60
                time_string = f"{minutes:02d}:{seconds:02d}"
                self.root.after(10, lambda t=time_string: self.time_value.config(text=t))
            time.sleep(1)
    
    def end_exercise_session(self):
        if self.camera_active:
            self.stop_camera()
            
            # Calculate session stats
            if self.session_start_time:
                duration = int(time.time() - self.session_start_time)
                
                # Save session
                self.save_session(duration)
                
                # Show summary
                self.show_session_summary(duration)
    
    def save_session(self, duration):
        if not self.session_angles:
            return
            
        # Calculate performance score based on how well angles match target
        exercise = self.exercise_data[self.current_exercise]
        target = exercise["target_angles"]["ideal"]
        max_deviation = max(target - exercise["target_angles"]["min"], 
                          exercise["target_angles"]["max"] - target)
        
        deviations = []
        for angle in self.session_angles:
            deviation = abs(angle - target) / max_deviation
            deviations.append(min(1.0, deviation))
        
        avg_deviation = sum(deviations) / len(deviations)
        performance = int((1 - avg_deviation) * 100)
        
        # Create session record
        session = {
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "exercise": exercise["name"],
            "duration": duration // 60,  # Convert to minutes
            "reps": self.session_reps,
            "performance": performance,
            "angles": self.session_angles[:100]  # Limit to 100 records
        }
        
        # Load existing activities
        activities = self.load_recent_activities()
        activities.append(session)
        
        # Save activities (keep only latest 20)
        with open(os.path.join(self.app_dir, "data", "activities.json"), "w") as f:
            json.dump(activities[-20:], f)
    
    def show_session_summary(self, duration):
        # Create popup window
        summary = tk.Toplevel(self.root)
        summary.title("Session Summary")
        summary.geometry("600x500")
        summary.configure(bg=self.theme["bg_main"])
        summary.transient(self.root)
        summary.grab_set()
        
        # Add header
        header_frame = tk.Frame(summary, bg=self.theme["bg_main"], padx=20, pady=20)
        header_frame.pack(fill='x')
        
        header_label = tk.Label(header_frame, text="Session Complete!", font=self.heading_font,
                              bg=self.theme["bg_main"], fg=self.theme["text_primary"])
        header_label.pack(anchor="w")
        
        exercise = self.exercise_data[self.current_exercise]["name"]
        subheader_label = tk.Label(header_frame, text=f"Exercise: {exercise}", font=self.subheading_font,
                                 bg=self.theme["bg_main"], fg=self.theme["text_primary"])
        subheader_label.pack(anchor="w")
        
        # Add stats
        stats_frame = tk.Frame(summary, bg=self.theme["bg_main"], padx=20, pady=10)
        stats_frame.pack(fill='x')
        
        stats_label = tk.Label(stats_frame, text="Session Statistics", font=self.subheading_font,
                             bg=self.theme["bg_main"], fg=self.theme["text_primary"])
        stats_label.pack(anchor="w", pady=(0, 10))
        
        # Create 2x2 grid for stats
        grid_frame = tk.Frame(stats_frame, bg=self.theme["bg_main"])
        grid_frame.pack(fill='x')
        
        # Duration
        duration_frame = tk.Frame(grid_frame, bg=self.theme["bg_secondary"], padx=15, pady=15)
        duration_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        duration_label = tk.Label(duration_frame, text="Duration", font=self.normal_font,
                                bg=self.theme["bg_secondary"], fg=self.theme["text_secondary"])
        duration_label.pack(anchor="w")
        
        minutes = duration // 60
        seconds = duration % 60
        time_string = f"{minutes}m {seconds}s"
        duration_value = tk.Label(duration_frame, text=time_string, font=self.subheading_font,
                                bg=self.theme["bg_secondary"], fg=self.theme["text_primary"])
        duration_value.pack(anchor="w")
        
        # Repetitions
        reps_frame = tk.Frame(grid_frame, bg=self.theme["bg_secondary"], padx=15, pady=15)
        reps_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        reps_label = tk.Label(reps_frame, text="Repetitions", font=self.normal_font,
                            bg=self.theme["bg_secondary"], fg=self.theme["text_secondary"])
        reps_label.pack(anchor="w")
        
        reps_value = tk.Label(reps_frame, text=str(self.session_reps), font=self.subheading_font,
                            bg=self.theme["bg_secondary"], fg=self.theme["text_primary"])
        reps_value.pack(anchor="w")
        
        # Performance
        if self.session_angles:
            exercise = self.exercise_data[self.current_exercise]
            target = exercise["target_angles"]["ideal"]
            max_deviation = max(target - exercise["target_angles"]["min"], 
                              exercise["target_angles"]["max"] - target)
            
            deviations = []
            for angle in self.session_angles:
                deviation = abs(angle - target) / max_deviation
                deviations.append(min(1.0, deviation))
            
            avg_deviation = sum(deviations) / len(deviations)
            performance = int((1 - avg_deviation) * 100)
            
            perf_frame = tk.Frame(grid_frame, bg=self.theme["bg_secondary"], padx=15, pady=15)
            perf_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
            
            perf_label = tk.Label(perf_frame, text="Performance", font=self.normal_font,
                                bg=self.theme["bg_secondary"], fg=self.theme["text_secondary"])
            perf_label.pack(anchor="w")
            
            perf_value = tk.Label(perf_frame, text=f"{performance}%", font=self.subheading_font,
                                bg=self.theme["bg_secondary"], fg=self.theme["text_primary"])
            perf_value.pack(anchor="w")
            
            # Common feedback
            feedback_counts = {}
            for feedback in self.session_feedback:
                if feedback in feedback_counts:
                    feedback_counts[feedback] += 1
                else:
                    feedback_counts[feedback] = 1
                    
            most_common = max(feedback_counts.items(), key=lambda x: x[1])
            
            feedback_frame = tk.Frame(grid_frame, bg=self.theme["bg_secondary"], padx=15, pady=15)
            feedback_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
            
            feedback_label = tk.Label(feedback_frame, text="Most Common Feedback", font=self.normal_font,
                                    bg=self.theme["bg_secondary"], fg=self.theme["text_secondary"])
            feedback_label.pack(anchor="w")
            
            feedback_value = tk.Label(feedback_frame, text=most_common[0], font=self.subheading_font,
                                    bg=self.theme["bg_secondary"], fg=self.theme["text_primary"])
            feedback_value.pack(anchor="w")
            
            # Add angle chart
            chart_frame = tk.Frame(summary, bg=self.theme["bg_main"], padx=20, pady=10)
            chart_frame.pack(fill='both', expand=True)
            
            chart_label = tk.Label(chart_frame, text="Angle Progression", font=self.subheading_font,
                                 bg=self.theme["bg_main"], fg=self.theme["text_primary"])
            chart_label.pack(anchor="w", pady=(0, 10))
            
            # Create figure
            fig = plt.Figure(figsize=(5, 3), dpi=100)
            ax = fig.add_subplot(111)
            
            # Sample data points to avoid overcrowding
            sampled_angles = self.session_angles[::max(1, len(self.session_angles) // 50)]
            ax.plot(sampled_angles, color=exercise["color"])
            
            # Add target range
            ax.axhline(y=exercise["target_angles"]["ideal"], color='green', linestyle='--', alpha=0.7)
            ax.axhspan(exercise["target_angles"]["min"], exercise["target_angles"]["max"], 
                     alpha=0.2, color='green')
            
            ax.set_ylabel('Angle (degrees)')
            ax.set_xlabel('Time')
            ax.grid(True, alpha=0.3)
            
            # Add chart to frame
            canvas = FigureCanvasTkAgg(fig, master=chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Add buttons
        button_frame = tk.Frame(summary, bg=self.theme["bg_main"], padx=20, pady=20)
        button_frame.pack(fill='x')
        
        close_button = tk.Button(button_frame, text="Close", font=self.normal_font,
                               bg=self.theme["bg_tertiary"], fg=self.theme["text_primary"], padx=15, pady=8,
                               command=summary.destroy)
        close_button.pack(side="right")
        
        new_session_button = tk.Button(button_frame, text="New Session", font=self.normal_font,
                                     bg=self.theme["success"], fg="white", padx=15, pady=8,
                                     command=lambda: [summary.destroy(), self.show_exercise_selection()])
        new_session_button.pack(side="right", padx=10)
    
    def show_progress(self):
        self.current_page = "progress"
        self.clear_frame(self.content_frame)
        
        # Create header
        header_frame = tk.Frame(self.content_frame, bg=self.theme["bg_main"], padx=20, pady=20)
        header_frame.pack(fill='x')
        
        header_label = tk.Label(header_frame, text="Progress Tracker", font=self.heading_font,
                              bg=self.theme["bg_main"], fg=self.theme["text_primary"])
        header_label.pack(anchor="w")
        
        # Create charts frame
        charts_frame = tk.Frame(self.content_frame, bg=self.theme["bg_main"], padx=20, pady=10)
        charts_frame.pack(fill='both', expand=True)
        
        # Load activities
        activities = self.load_recent_activities()
        
        if not activities:
            no_data = tk.Label(charts_frame, text="No activity data available yet. Complete exercise sessions to see your progress.",
                             font=self.normal_font, bg=self.theme["bg_tertiary"], fg=self.theme["text_secondary"],
                             padx=15, pady=30)
            no_data.pack(fill='x', padx=5, pady=5)
            return
            
        # Create exercise tabs
        exercises = sorted(list(set([a["exercise"] for a in activities])))
        
        tabs_frame = tk.Frame(charts_frame, bg=self.theme["bg_main"])
        tabs_frame.pack(fill='x')
        
        # Track current selected tab
        self.selected_tab = tk.StringVar(value=exercises[0] if exercises else "")
        
        for exercise in exercises:
            tab_button = tk.Button(tabs_frame, text=exercise, font=self.normal_font,
                                 bg=self.theme["bg_tertiary"], fg=self.theme["text_primary"],
                                 padx=15, pady=8, bd=0,
                                 command=lambda e=exercise: self.change_progress_tab(e))
            tab_button.pack(side="left", padx=5, pady=5)
        
        # Add content frame for selected exercise
        self.progress_content = tk.Frame(charts_frame, bg=self.theme["bg_main"])
        self.progress_content.pack(fill='both', expand=True, pady=10)
        
        # Show first exercise by default
        if exercises:
            self.change_progress_tab(exercises[0])
    
    def change_progress_tab(self, exercise):
        self.selected_tab.set(exercise)
        self.clear_frame(self.progress_content)
        
        # Load activities for this exercise
        activities = self.load_recent_activities()
        exercise_activities = [a for a in activities if a["exercise"] == exercise]
        
        if not exercise_activities:
            return
            
        # Create stats summary
        stats_frame = tk.Frame(self.progress_content, bg=self.theme["bg_main"])
        stats_frame.pack(fill='x', pady=10)
        
        # Total sessions
        sessions_frame = tk.Frame(stats_frame, bg=self.theme["bg_secondary"], padx=15, pady=15)
        sessions_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        sessions_label = tk.Label(sessions_frame, text="Total Sessions", font=self.normal_font,
                                bg=self.theme["bg_secondary"], fg=self.theme["text_secondary"])
        sessions_label.pack(anchor="w")
        
        sessions_value = tk.Label(sessions_frame, text=str(len(exercise_activities)), font=self.subheading_font,
                                bg=self.theme["bg_secondary"], fg=self.theme["text_primary"])
        sessions_value.pack(anchor="w")
        
        # Total time
        time_frame = tk.Frame(stats_frame, bg=self.theme["bg_secondary"], padx=15, pady=15)
        time_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        time_label = tk.Label(time_frame, text="Total Time", font=self.normal_font,
                            bg=self.theme["bg_secondary"], fg=self.theme["text_secondary"])
        time_label.pack(anchor="w")
        
        total_time = sum([a.get("duration", 0) for a in exercise_activities])
        time_value = tk.Label(time_frame, text=f"{total_time} min", font=self.subheading_font,
                            bg=self.theme["bg_secondary"], fg=self.theme["text_primary"])
        time_value.pack(anchor="w")
        
        # Average performance
        if any("performance" in a for a in exercise_activities):
            perf_frame = tk.Frame(stats_frame, bg=self.theme["bg_secondary"], padx=15, pady=15)
            perf_frame.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
            
            perf_label = tk.Label(perf_frame, text="Avg Performance", font=self.normal_font,
                                bg=self.theme["bg_secondary"], fg=self.theme["text_secondary"])
            perf_label.pack(anchor="w")
            
            perfs = [a.get("performance", 0) for a in exercise_activities if "performance" in a]
            avg_perf = sum(perfs) / len(perfs) if perfs else 0
            perf_value = tk.Label(perf_frame, text=f"{int(avg_perf)}%", font=self.subheading_font,
                                bg=self.theme["bg_secondary"], fg=self.theme["text_primary"])
            perf_value.pack(anchor="w")
        
        # Create performance over time chart
        if any("performance" in a for a in exercise_activities):
            chart_frame = tk.Frame(self.progress_content, bg=self.theme["bg_main"], padx=15, pady=15)
            chart_frame.pack(fill='both', expand=True)
            
            chart_label = tk.Label(chart_frame, text="Performance History", font=self.subheading_font,
                                 bg=self.theme["bg_main"], fg=self.theme["text_primary"])
            chart_label.pack(anchor="w", pady=(0, 10))
            
            # Create figure
            fig = plt.Figure(figsize=(8, 4), dpi=100)
            ax = fig.add_subplot(111)
            
            # Get data
            dates = [a["date"].split()[0] for a in exercise_activities if "performance" in a]
            performances = [a.get("performance", 0) for a in exercise_activities if "performance" in a]
            
            # Plot
            if dates and performances:
                ax.plot(dates, performances, marker='o', color=self.theme["accent"])
                ax.set_ylabel('Performance (%)')
                ax.set_xlabel('Date')
                ax.grid(True, alpha=0.3)
                
                # Rotate date labels
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                
                # Add chart to frame
                canvas = FigureCanvasTkAgg(fig, master=chart_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Create recent sessions list
        sessions_frame = tk.Frame(self.progress_content, bg=self.theme["bg_main"], padx=15, pady=15)
        sessions_frame.pack(fill='x')
        
        sessions_label = tk.Label(sessions_frame, text="Recent Sessions", font=self.subheading_font,
                                bg=self.theme["bg_main"], fg=self.theme["text_primary"])
        sessions_label.pack(anchor="w", pady=(0, 10))
        
        # Create list frame
        list_frame = tk.Frame(sessions_frame, bg=self.theme["bg_main"])
        list_frame.pack(fill='x')
        
        # Headers
        headers = ["Date", "Duration", "Performance"]
        header_frame = tk.Frame(list_frame, bg=self.theme["bg_tertiary"])
        header_frame.pack(fill='x')
        
        for i, header in enumerate(headers):
            header_label = tk.Label(header_frame, text=header, font=self.normal_font,
                                  bg=self.theme["bg_tertiary"], fg=self.theme["text_secondary"],
                                  padx=15, pady=10)
            header_label.grid(row=0, column=i, sticky="w")
        
        # List items
        for i, activity in enumerate(exercise_activities[:10]):  # Show only last 10 activities
            item_frame = tk.Frame(list_frame, bg=self.theme["bg_main"] if i % 2 == 0 else self.theme["bg_secondary"])
            item_frame.pack(fill='x')
            
            date_label = tk.Label(item_frame, text=activity["date"], font=self.normal_font,
                                bg=item_frame["bg"], fg=self.theme["text_primary"], padx=15, pady=10)
            date_label.grid(row=0, column=0, sticky="w")
            
            duration_label = tk.Label(item_frame, text=f"{activity.get('duration', 0)} min", font=self.normal_font,
                                    bg=item_frame["bg"], fg=self.theme["text_primary"], padx=15, pady=10)
            duration_label.grid(row=0, column=1, sticky="w")
            
            if "performance" in activity:
                perf_label = tk.Label(item_frame, text=f"{activity['performance']}%", font=self.normal_font,
                                    bg=item_frame["bg"], fg=self.theme["text_primary"], padx=15, pady=10)
                perf_label.grid(row=0, column=2, sticky="w")
    
    def show_settings(self):
        self.current_page = "settings"
        self.clear_frame(self.content_frame)
        
        # Create header
        header_frame = tk.Frame(self.content_frame, bg=self.theme["bg_main"], padx=20, pady=20)
        header_frame.pack(fill='x')
        
        header_label = tk.Label(header_frame, text="Settings", font=self.heading_font,
                              bg=self.theme["bg_main"], fg=self.theme["text_primary"])
        header_label.pack(anchor="w")
        
        # Create settings frame
        settings_frame = tk.Frame(self.content_frame, bg=self.theme["bg_main"], padx=20, pady=10)
        settings_frame.pack(fill='both')
        
        # User profile section
        profile_frame = tk.Frame(settings_frame, bg=self.theme["bg_secondary"], padx=20, pady=20)
        profile_frame.pack(fill='x', pady=10)
        
        profile_label = tk.Label(profile_frame, text="User Profile", font=self.subheading_font,
                               bg=self.theme["bg_secondary"], fg=self.theme["text_primary"])
        profile_label.pack(anchor="w", pady=(0, 10))
        
        # User name
        name_frame = tk.Frame(profile_frame, bg=self.theme["bg_secondary"], pady=5)
        name_frame.pack(fill='x')
        
        name_label = tk.Label(name_frame, text="Display Name:", font=self.normal_font,
                            bg=self.theme["bg_secondary"], fg=self.theme["text_secondary"], width=15, anchor="w")
        name_label.pack(side="left")
        
        name_entry = ttk.Entry(name_frame, font=self.normal_font)
        name_entry.insert(0, self.current_user)
        name_entry.pack(side="left", fill='x', expand=True)
        
        # Save profile button
        save_button = tk.Button(profile_frame, text="Save Profile", font=self.normal_font,
                              bg=self.theme["accent"], fg="white", padx=15, pady=5,
                              command=lambda: self.save_user_profile(name_entry.get()))
        save_button.pack(anchor="e", pady=(10, 0))
        
        # App settings section
        app_frame = tk.Frame(settings_frame, bg=self.theme["bg_secondary"], padx=20, pady=20)
        app_frame.pack(fill='x', pady=10)
        
        app_label = tk.Label(app_frame, text="Application Settings", font=self.subheading_font,
                           bg=self.theme["bg_secondary"], fg=self.theme["text_primary"])
        app_label.pack(anchor="w", pady=(0, 10))
        
        # Dark mode toggle
        theme_frame = tk.Frame(app_frame, bg=self.theme["bg_secondary"], pady=5)
        theme_frame.pack(fill='x')
        
        theme_label = tk.Label(theme_frame, text="Dark Mode:", font=self.normal_font,
                             bg=self.theme["bg_secondary"], fg=self.theme["text_secondary"], width=15, anchor="w")
        theme_label.pack(side="left")
        
        theme_toggle = ttk.Checkbutton(theme_frame, variable=self.dark_mode, command=self.toggle_theme)
        theme_toggle.pack(side="left")
        
        # Voice feedback toggle
        voice_frame = tk.Frame(app_frame, bg=self.theme["bg_secondary"], pady=5)
        voice_frame.pack(fill='x')
        
        voice_label = tk.Label(voice_frame, text="Voice Feedback:", font=self.normal_font,
                             bg=self.theme["bg_secondary"], fg=self.theme["text_secondary"], width=15, anchor="w")
        voice_label.pack(side="left")
        
        voice_toggle = ttk.Checkbutton(voice_frame, variable=self.use_voice_feedback)
        voice_toggle.pack(side="left")
        
        # Camera settings
        camera_frame = tk.Frame(settings_frame, bg=self.theme["bg_secondary"], padx=20, pady=20)
        camera_frame.pack(fill='x', pady=10)
        
        camera_label = tk.Label(camera_frame, text="Camera Settings", font=self.subheading_font,
                              bg=self.theme["bg_secondary"], fg=self.theme["text_primary"])
        camera_label.pack(anchor="w", pady=(0, 10))
        
        # Camera selection
        cam_frame = tk.Frame(camera_frame, bg=self.theme["bg_secondary"], pady=5)
        cam_frame.pack(fill='x')
        
        cam_label = tk.Label(cam_frame, text="Camera ID:", font=self.normal_font,
                           bg=self.theme["bg_secondary"], fg=self.theme["text_secondary"], width=15, anchor="w")
        cam_label.pack(side="left")
        
        cam_value = tk.StringVar(value=str(self.camera_id))
        cam_entry = ttk.Entry(cam_frame, textvariable=cam_value, font=self.normal_font, width=5)
        cam_entry.pack(side="left")
        
        cam_test = tk.Button(cam_frame, text="Test Camera", font=self.normal_font,
                           bg=self.theme["bg_tertiary"], fg=self.theme["text_primary"], padx=15, pady=5,
                           command=lambda: self.test_camera(int(cam_value.get())))
        cam_test.pack(side="left", padx=10)
        
        # Calibration button
        calibrate_button = tk.Button(camera_frame, text="Calibrate Camera", font=self.normal_font,
                                   bg=self.theme["accent"], fg="white", padx=15, pady=5,
                                   command=self.start_calibration)
        calibrate_button.pack(anchor="e", pady=(10, 0))
        
        # Data management section
        data_frame = tk.Frame(settings_frame, bg=self.theme["bg_secondary"], padx=20, pady=20)
        data_frame.pack(fill='x', pady=10)
        
        data_label = tk.Label(data_frame, text="Data Management", font=self.subheading_font,
                            bg=self.theme["bg_secondary"], fg=self.theme["text_primary"])
        data_label.pack(anchor="w", pady=(0, 10))
        
        # Export data button
        export_button = tk.Button(data_frame, text="Export Exercise Data", font=self.normal_font,
                                bg=self.theme["bg_tertiary"], fg=self.theme["text_primary"], padx=15, pady=5,
                                command=self.export_data)
        export_button.pack(anchor="w")
        
        # Clear data button
        clear_button = tk.Button(data_frame, text="Clear All Data", font=self.normal_font,
                               bg=self.theme["error"], fg="white", padx=15, pady=5,
                               command=self.clear_data)
        clear_button.pack(anchor="w", pady=10)
        
        # About section
        about_frame = tk.Frame(settings_frame, bg=self.theme["bg_secondary"], padx=20, pady=20)
        about_frame.pack(fill='x', pady=10)
        
        about_label = tk.Label(about_frame, text="About", font=self.subheading_font,
                             bg=self.theme["bg_secondary"], fg=self.theme["text_primary"])
        about_label.pack(anchor="w", pady=(0, 10))
        
        version_label = tk.Label(about_frame, text="Exercise Posture Pro - Version 2.0", font=self.normal_font,
                               bg=self.theme["bg_secondary"], fg=self.theme["text_primary"])
        version_label.pack(anchor="w")
        
        desc_label = tk.Label(about_frame, text="Advanced exercise form correction system using computer vision",
                            font=self.small_font, bg=self.theme["bg_secondary"], fg=self.theme["text_secondary"],
                            wraplength=600, justify="left")
        desc_label.pack(anchor="w", pady=(5, 10))
        
        # Add links
        help_button = tk.Button(about_frame, text="Help & Documentation", font=self.normal_font,
                              bg=self.theme["bg_tertiary"], fg=self.theme["text_primary"], padx=15, pady=5,
                              command=lambda: webbrowser.open("https://example.com/help"))
        help_button.pack(anchor="w")
    
    def test_camera(self, camera_id):
        # Test if camera is accessible
        try:
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                messagebox.showerror("Error", f"Could not open camera ID {camera_id}")
                return
                
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Error", f"Could not read from camera ID {camera_id}")
                cap.release()
                return
                
            # Camera works, update setting
            self.camera_id = camera_id
            cap.release()
            messagebox.showinfo("Success", f"Camera ID {camera_id} is working properly")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error testing camera: {str(e)}")
    
    def start_calibration(self):
        messagebox.showinfo("Calibration", "Camera calibration feature coming soon!")
    
    def export_data(self):
        # Ask for export location
        export_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Export Exercise Data"
        )
        
        if not export_path:
            return
            
        # Export data
        try:
            activities = self.load_recent_activities()
            with open(export_path, "w") as f:
                json.dump(activities, f, indent=4)
                
            messagebox.showinfo("Success", f"Data exported successfully to {export_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error exporting data: {str(e)}")
    
    def clear_data(self):
        # Confirm before clearing
        confirm = messagebox.askyesno("Confirm", "Are you sure you want to clear all activity data? This cannot be undone.")
        
        if confirm:
            try:
                data_path = os.path.join(self.app_dir, "data", "activities.json")
                if os.path.exists(data_path):
                    os.remove(data_path)
                    
                messagebox.showinfo("Success", "All activity data has been cleared")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error clearing data: {str(e)}")
    
    def show_help(self):
        self.current_page = "help"
        self.clear_frame(self.content_frame)
        
        # Create header
        header_frame = tk.Frame(self.content_frame, bg=self.theme["bg_main"], padx=20, pady=20)
        header_frame.pack(fill='x')
        
        header_label = tk.Label(header_frame, text="Help & Documentation", font=self.heading_font,
                              bg=self.theme["bg_main"], fg=self.theme["text_primary"])
        header_label.pack(anchor="w")
        
        # Create help content
        help_frame = tk.Frame(self.content_frame, bg=self.theme["bg_main"], padx=20, pady=10)
        help_frame.pack(fill='both', expand=True)
        
        # Getting started section
        start_frame = tk.Frame(help_frame, bg=self.theme["bg_secondary"], padx=20, pady=20)
        start_frame.pack(fill='x', pady=10)
        
        start_label = tk.Label(start_frame, text="Getting Started", font=self.subheading_font,
                             bg=self.theme["bg_secondary"], fg=self.theme["text_primary"])
        start_label.pack(anchor="w", pady=(0, 10))
        
        steps = [
            "1. Select an exercise from the Dashboard or Exercise Selection page",
            "2. Position yourself in front of the camera so your full body is visible",
            "3. Click 'Start Camera' to begin exercise tracking",
            "4. Follow the on-screen feedback to improve your form",
            "5. Complete your exercise session and view your performance summary"
        ]
        
        for step in steps:
            step_label = tk.Label(start_frame, text=step, font=self.normal_font,
                                bg=self.theme["bg_secondary"], fg=self.theme["text_primary"],
                                wraplength=800, justify="left")
            step_label.pack(anchor="w", pady=5)
        
        # Exercise guide section
        guide_frame = tk.Frame(help_frame, bg=self.theme["bg_secondary"], padx=20, pady=20)
        guide_frame.pack(fill='x', pady=10)
        
        guide_label = tk.Label(guide_frame, text="Exercise Guide", font=self.subheading_font,
                             bg=self.theme["bg_secondary"], fg=self.theme["text_primary"])
        guide_label.pack(anchor="w", pady=(0, 10))
        
        exercises = [
            "Bicep Curl - Stand with arms extended, bend at the elbow to lift weights toward shoulders",
            "Push-up - Start in plank position, lower body until elbows reach 90°, then push back up",
            "Squat - Stand with feet shoulder-width apart, bend knees to lower body, keep back straight",
            "Shoulder Press - Start with weights at shoulder height, push directly upward until arms extend"
        ]
        
        for ex in exercises:
            ex_label = tk.Label(guide_frame, text=ex, font=self.normal_font,
                              bg=self.theme["bg_secondary"], fg=self.theme["text_primary"],
                              wraplength=800, justify="left")
            ex_label.pack(anchor="w", pady=5)
        
        # Troubleshooting section
        trouble_frame = tk.Frame(help_frame, bg=self.theme["bg_secondary"], padx=20, pady=20)
        trouble_frame.pack(fill='x', pady=10)
        
        trouble_label = tk.Label(trouble_frame, text="Troubleshooting", font=self.subheading_font,
                               bg=self.theme["bg_secondary"], fg=self.theme["text_primary"])
        trouble_label.pack(anchor="w", pady=(0, 10))
        
        issues = [
            "Camera not working - Check camera ID in Settings and ensure no other application is using the camera",
            "Poor pose detection - Ensure good lighting and that your full body is visible in the frame",
            "Inaccurate angle measurement - Try the camera calibration feature in Settings",
            "Application running slowly - Close other applications that might be using system resources"
        ]
        
        for issue in issues:
            issue_label = tk.Label(trouble_frame, text=issue, font=self.normal_font,
                                 bg=self.theme["bg_secondary"], fg=self.theme["text_primary"],
                                 wraplength=800, justify="left")
            issue_label.pack(anchor="w", pady=5)
        
        # Contact section
        contact_frame = tk.Frame(help_frame, bg=self.theme["bg_secondary"], padx=20, pady=20)
        contact_frame.pack(fill='x', pady=10)
        
        contact_label = tk.Label(contact_frame, text="Contact & Support", font=self.subheading_font,
                               bg=self.theme["bg_secondary"], fg=self.theme["text_primary"])
        contact_label.pack(anchor="w", pady=(0, 10))
        
        email_label = tk.Label(contact_frame, text="For support, email: support@exerciseposturepro.com",
                             font=self.normal_font, bg=self.theme["bg_secondary"], fg=self.theme["text_primary"])
        email_label.pack(anchor="w", pady=5)
        
        website_label = tk.Label(contact_frame, text="Visit our website: www.exerciseposturepro.com",
                               font=self.normal_font, bg=self.theme["bg_secondary"], fg=self.theme["text_primary"])
        website_label.pack(anchor="w", pady=5)
    
    def save_user_profile(self, name):
        if name.strip():
            self.current_user = name.strip()
            
            # Save user profile
            with open(os.path.join(self.app_dir, "profiles", "user.json"), "w") as f:
                json.dump({"name": self.current_user}, f)
                
            messagebox.showinfo("Success", "User profile saved successfully")
            
            # Update dashboard
            if self.current_page == "dashboard":
                self.show_dashboard()
    
    def load_user_data(self):
        # Load user profile
        try:
            profile_path = os.path.join(self.app_dir, "profiles", "user.json")
            if os.path.exists(profile_path):
                with open(profile_path, "r") as f:
                    profile = json.load(f)
                    self.current_user = profile.get("name", "Default User")
        except:
            pass
    
    def load_recent_activities(self):
        try:
            activities_path = os.path.join(self.app_dir, "data", "activities.json")
            if os.path.exists(activities_path):
                with open(activities_path, "r") as f:
                    return json.load(f)
        except:
            pass
        return []
    
    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

# Main entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = ExercisePoseApp(root)
    root.mainloop()