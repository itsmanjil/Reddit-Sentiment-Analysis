import React, { useContext, useState } from "react";
import { useNavigate, Link, NavLink } from "react-router-dom";
import axiosInstance from "../../axios";
import { HashLink } from "react-router-hash-link";
import axios from "axios";
import AuthContext from "../../context/AuthContext";
// import Navbar from "../../Components/Navbar";

function Search() {
  function logoutHandler() {
    // console.log("logout");
    logoutUser();
  }

  const token = localStorage.getItem("authToken");
  console.log("token", token);
  const navigate = useNavigate();
  const [hasError, setHasError] = useState(false);
  const { logoutUser, authToken } = useContext(AuthContext);
  const [video_url, setVideoUrl] = useState("");
  const [max_comments, setMaxComments] = useState(200);
  const [use_api, setUseApi] = useState(false);
  const [sentimentModel, setSentimentModel] = useState("vader");
  const [isLoading, setIsLoading] = useState(false);
  const [searchError, setSearchError] = useState(false);
  const searchHandler = async (e) => {
    e.preventDefault();
    console.log("Analyze button clicked");
    if (!video_url) {
      console.log("Empty video URL");
      setHasError(true);
      navigate("/search");
    } else {
      try {
        setIsLoading(true);
        setHasError(false);
        const resp = await axios({
          method: "POST",
          url: `http://127.0.0.1:8000/api/youtube/analyze/`,
          timeout: 1000 * 180,
          validateStatus: (status) => {
            return status < 500;
          },
          data: {
            video_url: video_url,
            max_comments: max_comments,
            use_api: use_api,
            sentiment_model: sentimentModel,
          },
          headers: {
            Authorization: authToken
              ? "Bearer " + String(authToken.access)
              : null,
            "Content-Type": "application/json",
            accept: "application/json",
          },
        });
        console.log("YouTube analysis response:", resp.data);
        setIsLoading(false);
        navigate("/dashboard", {
          state: resp.data,
        });
      } catch (e) {
        if (e.response && e.response.status === 500) {
          setSearchError(true);
        }
        console.error("Analysis error:", e);
        setIsLoading(false);
      }
    }
  };
  return (
    <>
      <nav
        id="navbarExample"
        className="navbar navbar-expand-lg fixed-top"
        aria-label="Main navigation"
      >
        <div className="container">
          {/* <!-- Image Logo --> */}
          <Link to="/" className="navbar-brand logo-image">
            <img
              src="../assets/img/logo2.png"
              alt="alternative"
              style={{ height: "40px", width: "40px" }}
            />
          </Link>
          <Link to="/" className="navbar-brand logo-text">
            YouTube Sentiment
          </Link>
          <button
            className="navbar-toggler p-0 border-0"
            type="button"
            id="navbarSideCollapse"
            aria-label="Toggle navigation"
          >
            <span className="navbar-toggler-icon"></span>
          </button>

          <div
            className="navbar-collapse offcanvas-collapse"
            id="navbarsExampleDefault"
          >
            <ul className="navbar-nav ms-auto navbar-nav-scroll">
              <li className="nav-item">
                <Link to="/" className="nav-link" aria-current="page">
                  Home
                </Link>
              </li>
              

              {token !== null && (
                <>
                  <li className="nav-item">
                    <Link to="/dashboard" className="nav-link" aria-current="page">
                      Dashboard
                    </Link>
                  </li>
                  <li className="nav-item">
                    <Link to="/profile" className="nav-link" aria-current="page">
                      Profile
                    </Link>
                  </li>

                  <li
                    className="nav-item"
                    style={{ color: "pointer" }}
                    onClick={logoutHandler}
                  >
                    <div className="nav-link" style={{ cursor: "pointer" }}>
                      
                      <div>
                        <span className="nav-link-text ms-1">Logout</span>
                      </div>
                    </div>
                  </li>
                </>
              )}
            </ul>
            
          </div>
        </div>
      </nav>
      <header className="ex-header">
        <div className="container">
          <div className="row">
            <div className="col-xl-10 offset-xl-1">
              <h1 className="text-center">Analyze YouTube Video</h1>
            </div>
          </div>
        </div>
      </header>
      <div className="container rounded bg-white mt-5 mb-5">
        <div className="row">
          <div className="col-md-3 ">
            
          </div>
          <div className="col-md-5 ">
            <div className="p-3 py-5">
              <div className="d-flex justify-content-between align-items-center mb-3">
                <h4 className="text-right">Video Analysis Settings</h4>
              </div>
              <form>
                {searchError && !isLoading && (
                  <p style={{ color: "red" }}>
                    Error analyzing video. Please check the URL and try again!
                  </p>
                )}
                <div className="row mt-3">
                  <div className="col-md-12">
                    <label className="labels">YouTube Video URL</label>
                    <input
                      name="video_url"
                      type="text"
                      className="form-control"
                      placeholder="https://www.youtube.com/watch?v=..."
                      value={video_url}
                      onChange={(e) => {
                        setVideoUrl(e.target.value);
                      }}
                      required
                    />
                  </div>
                  <div className="col-md-12 mt-3">
                    <label className="labels">Max Comments (1-1000)</label>
                    <input
                      name="max_comments"
                      type="number"
                      className="form-control"
                      placeholder="200"
                      min="1"
                      max="1000"
                      value={max_comments}
                      onChange={(e) => {
                        setMaxComments(parseInt(e.target.value) || 200);
                      }}
                    />
                  </div>
                  <div className="col-md-12 mt-3">
                    <label className="labels">
                      <input
                        type="checkbox"
                        checked={use_api}
                        onChange={(e) => setUseApi(e.target.checked)}
                        style={{ marginRight: "8px" }}
                      />
                      Use YouTube API (faster, requires API key)
                    </label>
                    <p style={{ fontSize: "12px", color: "#666", marginTop: "4px" }}>
                      Uncheck to use scraper mode (slower but no API key needed)
                    </p>
                  </div>
                  <div className="col-md-12 mt-3">
                    <label className="labels">Sentiment Model</label>
                    <select
                      className="form-control"
                      value={sentimentModel}
                      onChange={(e) => setSentimentModel(e.target.value)}
                    >
                      <option value="vader">VADER (fast, recommended)</option>
                      <option value="roberta">RoBERTa (high accuracy, slower)</option>
                      <option value="tfidf">TF-IDF (legacy)</option>
                    </select>
                    <p style={{ fontSize: "12px", color: "#666", marginTop: "4px" }}>
                      RoBERTa is slower on CPU and needs extra backend deps.
                    </p>
                  </div>
                  {hasError && (
                    <p style={{ color: "red" }}>YouTube URL is required</p>
                  )}
                </div>

                <div className="mt-5 text-center">
                  <input
                    className="p-2 mb-2 bg-primary text-white w-45 my-4 mb-2"
                    // className="btn btn-primary profile-button"
                    type="button"
                    onClick={searchHandler}
                    value={isLoading ? `Analyzing...` : `Analyze Video`}
                    disabled={isLoading ? true : false}
                  ></input>
                </div>
              </form>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

export default Search;
