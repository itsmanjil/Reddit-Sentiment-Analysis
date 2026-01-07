import React from "react";
import { Link } from "react-router-dom";
import Navbar from "../../Components/Navbar";
import RegisterForm from "./RegisterForm";
function Register() {
  return (
    <>
    <header class="ex-header">
            <div class="container">
                <div class="row">
                    <div class="col-xl-10 offset-xl-1">
                        <h1 class="text-center">Sign Up</h1>
                    </div> 
                </div> 
            </div> 
      </header>
      <Navbar />
      
      <RegisterForm />
    </>
  );
}

export default Register;
