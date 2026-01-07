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
  const [product_name, setProductName] = useState("");
  const [company_name, setCompanyName] = useState("");
  const [keywords, setKeywords] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [searchError, setSearchError] = useState(false);
  const searchHandler = async (e) => {
    e.preventDefault();
    console.log("Search button clicked");
    if (!product_name || !company_name || !keywords) {
      console.log("Empty");
      setHasError(true);
      navigate("/search");
    } else {
      try {
        setIsLoading(true);
        setHasError(false);
        const resp = await axios({
          method: "POST",
          url: `http://127.0.0.1:8000/api/sentiment/search/`,
          timeout: 1000 * 150,
          validateStatus: (status) => {
            return status < 500;
          },
          data: {
            product_name: product_name,
            company_name: company_name,
            keywords: keywords,
          },
          headers: {
            Authorization: authToken
              ? "Bearer " + String(authToken.access)
              : null,
            "Content-Type": "application/json",
            accept: "application/json",
          },
        });
        // console.log(resp.data.predicted_data);
        setIsLoading(false);
        navigate("/dashboard", {
          state: resp.data.sentiment_data,
        });
      } catch (e) {
        if (e.response.status === 500) {
          setSearchError(true);
        }
        // logoutUser();
        setIsLoading(false);
      }
    }
  };
  return (
    <>
      <nav
        id="navbarExample"
        class="navbar navbar-expand-lg fixed-top"
        aria-label="Main navigation"
      >
        <div class="container">
          {/* <!-- Image Logo --> */}
          <Link to="/" class="navbar-brand logo-image">
            <img
              src="../assets/img/logo2.png"
              alt="alternative"
              style={{ height: "40px", width: "40px" }}
            />
          </Link>
          <Link to="/" class="navbar-brand logo-text">
            Gadget Reviews
          </Link>
          <button
            class="navbar-toggler p-0 border-0"
            type="button"
            id="navbarSideCollapse"
            aria-label="Toggle navigation"
          >
            <span class="navbar-toggler-icon"></span>
          </button>

          <div
            class="navbar-collapse offcanvas-collapse"
            id="navbarsExampleDefault"
          >
            <ul class="navbar-nav ms-auto navbar-nav-scroll">
              <li class="nav-item">
                <Link to="/" class="nav-link" aria-current="page">
                  Home
                </Link>
              </li>
              {/* <li class="nav-item">
                <HashLink smooth to="#features" class="nav-link">
                  Features
                </HashLink>
              </li>
              <li class="nav-item">
                <HashLink smooth to="#details" class="nav-link" href="#details">
                  Details
                </HashLink>
              </li> */}

              {token !== null && (
                <>
                  <li class="nav-item">
                    <Link to="/dashboard" class="nav-link" aria-current="page">
                      Dashboard
                    </Link>
                  </li>
                  <li class="nav-item">
                    <Link to="/profile" class="nav-link" aria-current="page">
                      Profile
                    </Link>
                  </li>

                  <li
                    className="nav-item"
                    style={{ color: "pointer" }}
                    onClick={logoutHandler}
                  >
                    <div className="nav-link" style={{ cursor: "pointer" }}>
                      {/* <div className="nav-link text-center me-2 d-flex align-items-center justify-content-center">
                <i className="material-icons opacity-10">logout</i>
              </div> */}
                      <div>
                        <span className="nav-link-text ms-1">Logout</span>
                      </div>
                    </div>
                  </li>
                </>
              )}
            </ul>
            {/* {token == null && (
              <span class="nav-item">
                <Link to="/signout" class="btn-outline-sm">
                  Log out
                </Link>
              </span>
            )} */}
          </div>
        </div>
      </nav>
      <header class="ex-header">
        <div class="container">
          <div class="row">
            <div class="col-xl-10 offset-xl-1">
              <h1 class="text-center">Search Gadget</h1>
            </div>
          </div>
        </div>
      </header>
      <div class="container rounded bg-white mt-5 mb-5">
        <div class="row">
          <div class="col-md-3 ">
            {/* <div class="d-flex flex-column align-items-center text-center p-3 py-5">
                            <img class="rounded-circle mt-5" width="150px" />src="https://st3.depositphotos.com/15648834/17930/v/600/depositphotos_179308454-stock-illustration-unknown-person-silhouette-glasses-profile.jpg" />
                            <span class ="font-weight-bold">Edogaru</span>
                            <span class ="text-black-50">edogaru @mail.com.my</span>
                            <span></span>
                        </div> */}
          </div>
          <div class="col-md-5 ">
            <div class="p-3 py-5">
              <div class="d-flex justify-content-between align-items-center mb-3">
                <h4 class="text-right">Search Criteria</h4>
              </div>
              <form>
                {searchError && !isLoading && (
                  <p style={{ color: "red" }}>
                    Error in fetching data. Try some other keywords!!
                  </p>
                )}
                <div class="row mt-3">
                  <div class="col-md-12">
                    <label class="labels">Product Name</label>
                    <input
                      name="product_name"
                      type="text"
                      class="form-control"
                      placeholder="enter product name"
                      value={product_name}
                      onChange={(e) => {
                        setProductName(e.target.value);
                      }}
                      required
                    />
                  </div>
                  <div class="col-md-12">
                    <label class="labels">Company Name</label>
                    <input
                      name="company_name"
                      type="text"
                      class="form-control"
                      placeholder="enter company name"
                      value={company_name}
                      onChange={(e) => {
                        setCompanyName(e.target.value);
                      }}
                    />
                  </div>
                  <div class="col-md-12">
                    <label class="labels">Relevant Keywords</label>
                    <input
                      name="keywords"
                      type="text"
                      class="form-control"
                      placeholder="enter relevant keywords"
                      value={keywords}
                      onChange={(e) => {
                        setKeywords(e.target.value);
                      }}
                    />
                  </div>
                  {hasError && (
                    <p style={{ color: "red" }}>All fields are required</p>
                  )}
                </div>

                <div class="mt-5 text-center">
                  <input
                    class="p-2 mb-2 bg-primary text-white w-45 my-4 mb-2"
                    // class="btn btn-primary profile-button"
                    type="button"
                    onClick={searchHandler}
                    value={isLoading ? `Searching...` : `Start Search`}
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
