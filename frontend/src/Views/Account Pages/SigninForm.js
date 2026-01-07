import { React, useState, useEffect, useContext } from "react";
import { Link } from "react-router-dom";
import AuthContext from "../../context/AuthContext";

const SigninForm = () => {
  let { loginUser } = useContext(AuthContext);

  const [inputData, setInputData] = useState({
    email: "",
    password: "",
  });

  const { email, password } = inputData;

  const { isError } = useContext(AuthContext);

  const inputHanlder = (e) => {
    setInputData({
      ...inputData,
      [e.target.name]: e.target.value,
    });
  };

  const submitHandler = async (e) => {
    e.preventDefault();
    loginUser(email, password);
  };

  return (
    <>
      <div class="ex-form-1 pt-5 pb-5">
        <div class="container">
          <div class="row">
            <div class="col-xl-6 offset-xl-3">
              <div class="text-box mt-5 mb-5">
                <p class="mb-4">
                  Don't have an account? Then please{" "}
                  <Link to="/register">
                    <span style={{ color: "#6168ff" }}>Sign up</span>
                  </Link>
                </p>

                <form role="form" class="text-start">
                  {isError && (
                    <p style={{ color: "red" }}>
                      Invalid username or password.
                    </p>
                  )}
                  <div class="input-group input-group-outline mb-3">
                    {/* <label class="form-label">Email</label> */}
                    <input
                      type="email"
                      placeholder="Email"
                      class="form-control"
                      name="email"
                      value={email}
                      onChange={inputHanlder}
                    />
                  </div>
                  <div class="input-group input-group-outline mb-3">
                    {/* <label class="form-label">Password</label> */}
                    <input
                      type="password"
                      placeholder="Password"
                      class="form-control"
                      name="password"
                      value={password}
                      onChange={inputHanlder}
                    />
                  </div>
                  {/* <div class="form-check form-switch d-flex align-items-center mb-3">
                    <input
                      class="form-check-input"
                      type="checkbox"
                      id="rememberMe"
                    />
                    <label class="form-check-label mb-0 ms-2" for="rememberMe">
                      Remember me
                    </label>
                  </div> */}
                  <div class="text-center">
                    <button
                      type="submit"
                      class="p-2 mb-2 bg-primary text-white w-100 my-4 mb-2"
                      // class="btn btn-primary w-100 my-4 mb-2"
                      onClick={submitHandler}
                    >
                      Log In
                    </button>
                  </div>
                  {/* <p class="mt-4 text-sm text-center">
                      Don't have an account?
                      <Link to="/register">Sign up</Link>
                    </p> */}
                </form>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default SigninForm;
