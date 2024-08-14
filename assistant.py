from groq import Groq 
from PIL import ImageGrab,Image
import google.generativeai as genai 
from openai import OpenAI
import cv2 
import pyperclip
import pyaudio

#using llama 3 as the main model 
groq_client = Groq(api_key='GROQ_API_KEY')
#using the gemini model for low latency image processing 
genai.configure  (api_key='GOOGLE_API_KEY')
openai_client = OpenAI(api_key='OPENAI_API_KEY')
web_cam = cv2.VideoCapture(1)

system_msg = (
    'you are a multimodel Ai voice assistant. Your user may or may not have attached a image which is either a photo or a webcam image.'
    'Any image have been processed into ahighly detailed text prompt that will be attached to their transcribed voice prompt.'
    'generate the most useful and factful response possible ,carefully considering all previous generated text in your response before.'
    'adding new tokens to the response . Do not expect or request image just use the context if added . '
    'use all the context of this conversation so you respond is relevant to the conversation, make your response clear and concise'
)

convo = [{'role':'system','context':system_msg}]
generation_config = {
    'temperature' : 0.7,
    'top_p' : 1,
    'top_k': 1,
    'max_output_tokens':2048
}
#truning the safety settimgs 
safety_settings  = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]


#loading the model 
model = genai.GenerativeModel('gemini-flash-1.5-model',
                              generation_config=generation_config,
                              safety_settings = safety_settings)

#making the groq to coversate instead if giving a single prompt
def groq_prompt(prompt,img_context):
    if img_context :
        prompt=f'User prompt : {prompt} \n\n  Image context : {img_context}'
        convo.append({'role' : 'user' , 'content': prompt})
    chat_completion = groq_client.chat.completions.create(messages=convo,model='llama-3-70b-8192')
    response = chat_completion.choices[0].message
    convo.append(response)
    return response.content

def function_call(prompt):
    system_msg = (
        'You are an ai function calling model,ypu will determine whether extracting the user clipboard content.'
        'taking a screenshot , capturing the webcame or writing a piece of code is the best for the ai assisant to respond. '
        'to the user prompt,webcam can be assumed to be a noraml laptop webcam facing the user. you will ' 
        'repond to with only one selection form the list that is ["extracted clipboard","take screenshot","play a youtube video","capture webcam","writing a piece of code if asked","None]'
        'response to the suggestion which were asked by thw user '
    )
    function_convo = [{'role' : 'user','content':system_msg},
                      {'role':'user','content':prompt}]
    chat_completion = groq_client.chat.completions.create(messages=convo , model='llama-3-70b-8192')
    response = chat_completion.choices[0].message
    return response.content
    
    
def take_screenshot(): 
    path = ''
    screenshot = ImageGrab.grab()
    rgb_screenshot = screenshot.convert('RGB')
    rgb_screenshot.save(path,quality=15)

def web_cam_capture():
    if not web_cam.isOpened():
        print('Eroor : camers isnot opened / not found')
        exit() 
        pzth = 'webcame.jpg'
        ret, frame = web_cam.read()
        cv2.imwrite(path,frame)

def get_clipboard_text():
    clipboard_content = pyperclip.paste()
    if isinstance(clipboard_content, str):
        return clipboard_content  
    else:
        print(' No clipboard text to copy')
        return None 
    
def vision_prompt(prompt,photo_path):
    img = Image.open(photo_path)
    prompt = (
        'you are the vision analysis ai that provide the semtantic meaning from images to provide context.'
        'to send to another AI that  create the response to the user, Do not respond as an Ai assistant.'
        'Instead take the user prompt and  try to extract all the meaningful information from the Image'
        'which is relevant to the user prompt.Then generate as much as objective data from the image for the AI'
        f'asssistant who will respond to the user . \n USER PROMPT : {prompt}'
    )
    response = model.generate_content([prompt,img])
    return response.text

def speak(text):
    player_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16,channels=1,rate=24000,output=True)
    stream_start = False
    
    with openai_client.audio.speech.with_streaming_response.create{
        model = 'tts-1'
    }
while True:
    prompt = input('User : ')
    call = function_call(prompt)
    
    if 'take screenshot' in call :
        print('takin screenshot')
        take_screenshot()
        visual_context = vision_prompt(prompt=prompt,photo_path=)
    elif 'capture webcam' in call :
        print('capturing webcam')
        web_cam_capture()
        visual_context = vision_prompt(prompt=prompt,photo_path=)
    elif 'extract clipboard ' in call :
        print('copying the clipbi=oard text')
        paste = get_clipboard_text()
        prompt = f'{prompt}\n\n Clipboard content  {paste}'
        visual_context = None
        
    response = groq_prompt(prompt=prompt,img_context=visual_context)
    print(response)
    speak(response)

    
    