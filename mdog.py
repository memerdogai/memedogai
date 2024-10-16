import time
import pyautogui
import random
import openai
from PIL import Image
import numpy as np
import sys 
import json
import requests
import cv2
import numpy as np
import sys
import hashlib
import moviepy.editor as mpe

time.sleep(5)

model_engine = "gpt-3.5-turbo"
api_key = '<API KEY FOR CHAT GPT>'

TWEET_SERVICE_URL = "https://pumpfunclub.com/tweets/?format=json"  # URL of the Django service
CHECK_INTERVAL = 5  # Time interval (in seconds) between each service call

# List of famous painters
painters = [
    "Leonardo da Vinci", "Vincent van Gogh", "Pablo Picasso", 
    "Claude Monet", "Rembrandt", "Salvador Dalí", 
    "Michelangelo", "Edgar Degas", "Frida Kahlo", 
    "Johannes Vermeer", "Paul Cézanne", "Henri Matisse"
]

writers = [
    "William Shakespeare", 
    "Jane Austen", 
    "Mark Twain", 
    "George Orwell", 
    "Ernest Hemingway", 
    "F. Scott Fitzgerald", 
    "Virginia Woolf", 
    "Leo Tolstoy", 
    "Charles Dickens", 
    "Gabriel García Márquez"
]

topics = [
    "Technology news",
    "Sports updates",
    "Health and fitness",
    "Weather forecast",
    "Financial markets",
    "Entertainment and movies",
    "Travel destinations",
    "Politics and current events",
    "Food and recipes",
    "Science discoveries",
    "Art and culture",
    "Cryptocurrency updates",
    "Online shopping trends",
    "New product releases",
    "Personal development tips",
    "Memes and internet culture",
    "Social media trends",
    "Gaming news",
    "Fashion and style",
    "Home improvement ideas",
    "Bitcoin price fluctuations",
    "Ethereum network upgrades",
    "Memecoin trends",
    "Crypto market crashes",
    "DeFi innovations",
    "NFT crazes",
    "Blockchain security breaches",
    "Crypto regulations",
    "Web3 development",
    "Crypto whales and market manipulation",
    "Crypto mining and energy consumption",
    "Token airdrops",
    "Crypto exchange hacks",
    "Metaverse integration with crypto",
    "Staking and yield farming",
    "Rug pulls and scams",
    "ICO failures",
    "DAOs",
    "Play-to-Earn gaming in crypto",
    "Crypto influencers and their predictions"
]



# Select a painter at random

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to create GIF with variable frame duration based on audio length
def create_gif(png_files, gif_output_path, frame_rate=24):
    # Calculate duration per frame in milliseconds
    duration_per_frame = 1000 / frame_rate
    
    # Load images
    images = [Image.open(png) for png in png_files]
    
    # Save images as GIF with the calculated duration per frame
    images[0].save(gif_output_path, save_all=True, append_images=images[1:], duration=duration_per_frame, loop=0)
    
    print(f"GIF saved at {gif_output_path}")

# Function to combine GIF and audio into a video
def create_video_with_audio(gif_path, audio_path, video_output_path):
    gif_clip = mpe.VideoFileClip(gif_path)
    audio_clip = mpe.AudioFileClip(audio_path)
    
    video_clip = gif_clip.set_audio(audio_clip)
    video_clip.write_videofile(video_output_path, codec="libx264", audio_codec="aac")
    print(f"Video with audio saved at {video_output_path}") 

def get_face_regions(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.5, minNeighbors=7)

    print(f"Detected {len(faces)} faces before aspect ratio check.")  # Debugging output

    # Initialize list to store valid faces
    valid_faces = []

    for (x, y, w, h) in faces:
        # Calculate the aspect ratio (width-to-height ratio)
        aspect_ratio = w / float(h)

        # Define the acceptable aspect ratio range for faces
        if 0.8 <= aspect_ratio <= 1.5:  # Face width should be between 80% and 150% of the height
            valid_faces.append((x, y, w, h))
        else:
            print(f"Rejected face at ({x}, {y}) with aspect ratio {aspect_ratio:.2f}")

    print(f"Detected {len(valid_faces)} faces after aspect ratio check.")  # Debugging output

    if len(valid_faces) == 0:
        raise Exception("No valid face detected after aspect ratio check!")

    return valid_faces  # Return valid detected faces


def apply_face_overlay(source_image, target_images):
    # Get all detected face regions in the source image
    face_regions = get_face_regions(source_image)


    if not face_regions:
        raise ValueError("No face regions detected in the source image")

    # Create a copy of the source image to overlay the target image
    result = source_image.copy()

    for (x, y, w, h) in face_regions:
        # Randomly select a target image from the list
        target_image_path = random.choice(target_images)
        target_image = cv2.imread(target_image_path, cv2.IMREAD_UNCHANGED)  # Load PNG with alpha channel

        if target_image is None:
            print(f"Failed to load target image: {target_image_path}")
            continue  # Skip if the image failed to load

        # Resize the target image to be larger than the detected face
        target_width = w
        target_height = int(h * 1.3)  # Increase height by 30% to cover the head
        target_resized = cv2.resize(target_image, (target_width, target_height))

        # Adjust the position for the target image so that it centers over the face
        y_offset = y - int((target_height - h) / 2)  # Center the target image over the detected face
        if y_offset < 0:
            y_offset = 0  # Prevent going out of bounds

        # Get the alpha channel from the target image
        alpha_channel = target_resized[:, :, 3] / 255.0  # Normalize alpha to [0, 1]
        overlay_color = target_resized[:, :, :3]  # Get the RGB channels

        # Get the region of interest in the result image
        roi = result[y_offset:y_offset + target_height, x:x + target_width]

        # Blend the overlay with the ROI
        for c in range(3):  # Loop over each color channel
            roi[:, :, c] = roi[:, :, c] * (1 - alpha_channel) + overlay_color[:, :, c] * alpha_channel

        # Place the blended ROI back into the result image
        result[y_offset:y_offset + target_height, x:x + target_width] = roi

    return result

def create_meme():
    # Set OpenAI API key for image generation
    openai.api_key = api_key

    # Get input from command line argument

    random_writer = random.choice(writers)
    selected_topic = random.choice(topics)
    content = "create a super ironic, intresting, controversial, paradoxical, funny meme about a useless fact on the topic of " + selected_topic + ", responde with image description and image caption, reponde with a json object where description and caption is defined as description and caption"
    content_role = "You are the writer named "  + random_writer + " but you are autistic  "
    # Call GPT-3 model
    response = openai.ChatCompletion.create(
        model=model_engine,
        messages=[
            {"role": "system", "content": content_role},
            {"role": "user", "content": content},
        ])

    # Get response from GPT-3
    message_gpt = response.choices[0]['message']['content']
    print("RESPONSE FROM GPT")
    print(message_gpt)
    print("RESPONSE FROM GPT DONE")

    # Parse GPT-3 response
    parsed_data = json.loads(message_gpt)

    print("Description:", parsed_data["description"])
    print("Caption:", parsed_data["caption"])

    random_painter = random.choice(painters)
    description = parsed_data["description"]
    description += " make sure human faces are clear and looking at the camera, make art in style of " + random_painter
    caption = parsed_data["caption"]


    names = ["50-cent", "alex-jones", "anderson-cooper", "andrew-tate", "andrew-yang", "angela-merkel", "angie", "anna-kendrick", "anthony-fauci", "antonio-banderas", "aoc", "ariana-grande", "arnold-schwarzenegger", "ben-affleck", "ben-shapiro", "bernie-sanders", "beyonce", "bill-clinton", "bill-gates", "bill-oreilly", "billie-eilish", "cardi-b", "casey-affleck", "charlamagne", "conor-mcgregor", "darth-vader", "demi-lovato", "dj-khaled", "donald-trump", "dr-dre", "dr-phil", "drake", "dwayne-johnson", "elizabeth-holmes", "ellen-degeneres", "elon-musk", "emma-watson", "gilbert-gottfried", "greta-thunberg", "grimes", "hillary-clinton", "jason-alexander", "jay-z", "jeff-bezos", "jerry-seinfeld", "jim-cramer", "joe-biden", "joe-rogan", "john-cena", "jordan-peterson", "justin-bieber", "justin-trudeau", "kamala-harris", "kanye-west", "kardashian", "kermit", "kevin-hart", "lex-fridman", "lil-wayne", "mark-zuckerberg", "martin-shkreli", "matt-damon", "matthew-mcconaughey", "mike-tyson", "morgan-freeman", "patrick-stewart", "paul-mccartney", "pokimane", "prince-harry", "rachel-maddow", "robert-downey-jr", "ron-desantis", "sam-altman", "samuel-jackson", "sbf", "scarlett-johansson", "sean-hannity", "snoop-dogg", "stephen-hawking", "taylor-swift", "tucker-carlson", "tupac", "warren-buffett", "will-smith", "william"]
    random_name = random.choice(names)

    # Fetch TTS audio from API
    caption_audio = caption.split("#", 1)[0]
    response = requests.request(
        method="POST",
        url="https://api.neets.ai/v1/tts",
        headers={
            "Content-Type": "application/json",
            "X-API-Key": "<API CODE FOR NEETS AI>"
        },
        json={
            "text": caption_audio,
            "voice_id": random_name,
            "params": {
                "model": "ar-diff-50k"
            }
        }
    )

    with open("neets_demo.mp3", "wb") as f:
        f.write(response.content)


    # Generate the image
    response = openai.Image.create(
        model="dall-e-3",
        prompt=description,
        size="1024x1024",
        n=1,
    )

    # Get the URL of the generated image
    image_url = response['data'][0]['url']

    # Download the image
    image_response = requests.get(image_url, stream=True)

    # Convert the response content to numpy array
    arr = np.asarray(bytearray(image_response.content), dtype=np.uint8)

    # Read the image with cv2
    image = cv2.imdecode(arr, -1)

    # Define font parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 2
    font_color = (255, 255, 255)
    line_type = cv2.LINE_AA
    caption_simple = 'mDOG'
    # Get the size of the text
    #text_size = cv2.getTextSize(caption, font, 1, font_thickness)[0]
    text_size = cv2.getTextSize(caption_simple, font, 1, font_thickness)[0]

    # Calculate the font scale based on the image width and text length
    font_scale = min(1, (image.shape[1] * 0.8) / text_size[0])

    # Recalculate the text size with the updated font scale
    text_size = cv2.getTextSize(caption_simple, font, font_scale, font_thickness)[0]

    # Recalculate text position with raised text by 30 pixels from the bottom
    text_position = ((image.shape[1] - text_size[0]) // 2, image.shape[0] - 50)  # 20 (previous) + 30

    # Define the background color
    background_color = (0, 0, 0)  # Black color

    # Calculate text position with a background
    text_bg_position = (text_position[0], text_position[1] + 5)  # Adjusted position to make space for the background

    # Get the size of the background rectangle
    bg_size = (text_size[0] + 10, text_size[1] + 20)  # Adjusted size to fit text

    # Add background rectangle to the image
    cv2.rectangle(image, text_bg_position, (text_bg_position[0] + bg_size[0], text_bg_position[1] - bg_size[1]), background_color, -1)

    # Add caption text to the image on top of the background
    cv2.putText(image, caption_simple, text_position, font, font_scale, font_color, font_thickness, lineType=line_type)

    # Save the image with caption
    cv2.imwrite("meme/meme.jpg", image)


    #source_img = cv2.imread('target_image.jpg')  # Replace with your source image path
    source_img = image
    target_images = [
        'memer_dog_head.png',  
        'memer_dog_head.png'
    ]

    print(f"Source Image Loaded: {source_img is not None}")

    try:
        result = apply_face_overlay(source_img, target_images)

        # Save the result to disk
        cv2.imwrite('meme/meme.jpg', result)

        # List of PNG files

        for i in range(3):
            # Generate the image
            print_no = 3 -i 
            response = openai.Image.create(
                model="dall-e-3",
                prompt="generate art of the number " + str(print_no) + " in the art style of " + random_painter,
                size="1024x1024",
                n=1,
            )

            # Get the URL of the generated image
            image_url = response['data'][0]['url']
            
            # Download the image
            image_response = requests.get(image_url, stream=True)
            
            # Save the image to a file
            with open(f"image_{i+1}.png", 'wb') as file:
                for chunk in image_response.iter_content(chunk_size=8192):
                    file.write(chunk)
            
            print(f"Image {i+1} downloaded and saved as image_{i+1}.png")

        png_files = ["image_1.png", "image_2.png", "image_3.png", "meme/meme.jpg"]  # List of your PNG files
        gif_output_path = "output.gif"
        audio_path = "neets_demo.mp3"  # Your audio file
        video_output_path = "meme/final_video.mp4"

        # Get audio duration using moviepy
        audio_clip = mpe.AudioFileClip(audio_path)
        audio_duration = audio_clip.duration  # Duration in seconds

        # Calculate frame duration
        num_frames = len(png_files)
        duration_per_frame = audio_duration / num_frames

        # Create GIF from PNG files with calculated frame duration
        create_gif(png_files, gif_output_path, duration_per_frame)

        # Create video from GIF and audio
        create_video_with_audio(gif_output_path, audio_path, video_output_path)

        print("Face overlay completed and saved as 'result_image.jpg'.")
    except Exception as e:
        print(str(e))
        raise Exception("No valid face detected after aspect ratio check!")


    return caption

def click_specific_spot(x, y):
    pyautogui.click(x, y)  # Click on the specified coordinates


# Function to generate response from ChatGPT
def generate_response():

    openai.api_key = api_key
    model_engine = "gpt-3.5-turbo" 
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "system", "content": "You are god posting on tweeter"},
            {"role": "user", "content": "create a new and funny x post that is only 100 characters long and would be completly new"},
        ])

    message_gpt = response.choices[0]['message']['content']
    print("RESPONSE FROM GPT")
    print(message_gpt)
    print("RESPONSE FROM GPT DONE")
    return message_gpt

# Main function
def main_send_tweet():
    try:
        # Scroll the mouse 

        # Get the coordinates of the spot you want to click
        # You can get this by running pyautogui.displayMousePosition() and hovering over the spot
        x = 100  # Example x-coordinate
        y = 109  # Example y-coordinate

        # Click on the specific spot
        click_specific_spot(x, y)
        time.sleep(5)

        x = 320  # Example x-coordinate
        y = 769  # Example y-coordinate

        # Click on the specific spot
        click_specific_spot(x, y)
        time.sleep(5)

        pyautogui.typewrite(create_meme() + " #memedog #mdog $mDOG")

        time.sleep(1)
        
        for _ in range(2):
                pyautogui.press('tab')  # Press Tab three times
                time.sleep(0.00005)  # Adjust delay as needed            
        pyautogui.press('enter')
        time.sleep(1)

        x = 682  # Example x-coordinate
        y = 288  # Example y-coordinate

        click_specific_spot(x, y)
        time.sleep(2)

        x = 1208  # Example x-coordinate
        y = 599  # Example y-coordinate

        click_specific_spot(x, y)
        time.sleep(15)

        for _ in range(5):
                pyautogui.press('tab')  # Press Tab three times
                time.sleep(0.00005)  # Adjust delay as needed            
        pyautogui.press('enter')
        time.sleep(5)        
        return True
        print("Automation completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        return False


def get_tweet_hash(tweet):
    """
    Generates a hash for a tweet to ensure uniqueness.
    """
    tweet_string = tweet.get('content', '') + str(tweet.get('id', ''))
    return hashlib.sha256(tweet_string.encode('utf-8')).hexdigest()

def fetch_tweets():
    """
    Fetch tweets from the service and return as a list of dictionaries.
    """
    response = requests.get(TWEET_SERVICE_URL)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching tweets: {response.status_code}")
        return []

def main():
    printed_hashes = set()  # Set to store the hashes of printed tweets

    while True:
        status = main_send_tweet()
        if status:
            time.sleep(CHECK_INTERVAL)  # Wait before checking again
        else:
            time.sleep(5)  # Wait before checking again

if __name__ == "__main__":

    for _ in range(100):
        main()
        random_number = random.uniform(30, 55)
        time.sleep(random_number)