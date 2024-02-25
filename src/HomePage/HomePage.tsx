import React from 'react';
import './HomePage.css';

const HomePage = (props: any) => {
    return (
        <div className="home-page">
            <video autoPlay loop muted className="background-video">
                <source src={"./rain.mp4"} type="video/mp4"/>
                Your browser does not support the video tag.
            </video>

            <span className="text-gradient">Under the Weather</span>
            <div className="button-container">
                <button onClick={() => props.onFormSwitch("login")}>Login</button>
                <button onClick={() => props.onFormSwitch("register")}>Register</button>
            </div>
        </div>
    );
};

export default HomePage;
