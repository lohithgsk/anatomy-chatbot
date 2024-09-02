import streamlit as st
from pathlib import Path

def main():
    # Set page configuration
    st.set_page_config(page_title="Anatomy 101", page_icon=":books:")

    # Sidebar
    with st.sidebar:
        st.subheader("Your Documents")

        # Preloaded files (you can replace these with actual file paths)
        preloaded_files = {
            "Anatomy_Basics.pdf": "https://drive.google.com/file/d/10wLiPf48SjQdYIM3_Z16d-sboRIauXGP/view?usp=drive_link",
            "Muscle_Structure.pdf": "https://drive.google.com/file/d/1fCUHEXMmLqy6uLVspj1T3OopWdExDn6j/view?usp=drive_link",
            "Skeletal_System.pdf": "https://drive.google.com/file/d/1N73KxQmCPMPiMwTqyZ4kfd5g8_WYlT4M/view?usp=drive_link"
        }

        # Display preloaded files as links
        st.write("Preloaded Files:")
        for file_name, file_path in preloaded_files.items():
            # Generate a link to open the PDF in a new tab
            file_link = f'<a href="{file_path}" target="_blank">{file_name}</a>'
            st.markdown(file_link, unsafe_allow_html=True)

        # File uploader for new PDFs
        uploaded_file = st.file_uploader("Upload your PDFs here", type=["pdf"])

        if uploaded_file is not None:
            st.write(f"Uploaded file: {uploaded_file.name}")
            st.button("Process")

    # Main content
    st.title("Anatomy 101")  # Title after the sidebar

    st.header("Welcome to Anatomy 101")
    st.text_input("Ask a question!?")

if __name__ == '__main__':
    main()
