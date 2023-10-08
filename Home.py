import requests
import streamlit as st
from streamlit_lottie import st_lottie



# Find more emojis here: https://www.webfx.com/tools/emoji-cheat-sheet/
#Define page config and sidebar
st.set_page_config(page_title="Fun Projects and Analysis", page_icon=":sagittarius:", layout="wide")
st.sidebar.success("Select a page above.")
#DD7E7

# ---- Return Lottie url ----#
def load_lottieurl(url):
	r = requests.get(url)
	if r.status_code != 200:
		return None
	return r.json()


#Use local css
def local_css(file_name):
	with open(file_name) as f:
		st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style/style.css")


# ---- Load Assets ----#
lottie_squirrel = load_lottieurl("https://lottie.host/304a1222-3c84-46e5-b5ea-bf0258dac888/MYbQdr9SUe.json")


# ---- Header Section ----#
with st.container():
	st.subheader("Hi, I'm Romain :wave:")
	st.title("A Data Analyst in the USA (specific, I know)")
	st.write("I like to make things and I like to learn things with a lot of overlap inbetween.")


# ----What I Do ----
with st.container():
	st.write("---")
	left_column, right_column = st.columns(2)
	with left_column:
		st.header("What I do")
		st.write("##")
		st.write(
		"I am creating projects using datasets I find interesting or just plain quirky. Data like the squirrel Census. If you\'re interested, check out the projects page to the left."
  		)
		st.write(
		"""
		A little more about me:
		- I'm a financial analyst at Swiss banking giant Credit Suisse (now UBS!), and have gotten to flex my data analytics skills on big financial data sets
		- I love learning and I hate tedium, so I'm always looking for fun new projects, tech, and hobbies, and I'm always trying to automate the boring stuff
		- I want to perform meaningful analysis, but I also just like seeing what I can do, even if the result doesn't matter
		- I'm terrible with colors and have to google if two colors go together or ask my artist wife if what I've created looks like vomit on a screen
  		"""
		)
with right_column:
	st_lottie(lottie_squirrel, height=300, key="squirrel")


# ---- Contact ----#
with st.container():
	st.write("---")
	st.header("Want to get in touch with me?")
	st.write("##")

	#Documentation: https://formsubmit.co/
	contact_form = """
	<form action="https://formsubmit.co/cromainbaker@gmail.com" method="POST">
		<input type="hidden" name="_captcha" value="false">
     		<input type="text" name="name" placeholder="Your name" required>
     		<input type="email" name="email" placeholder="Your email" required>
     		<textarea name="message" placeholder="Your message here" required></textarea>
     		<button type="submit">Send</button>
	</form>"""
	left_column, right_column = st.columns(2)
	with left_column:
		st.markdown(contact_form, unsafe_allow_html=True)
	with right_column:
		st.empty()