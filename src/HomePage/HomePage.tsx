import React from 'react';
import './HomePage.css';

const HomePage = (props: any) => {
    return (
        <div className="home-page">
            <img src={`${process.env.PUBLIC_URL}/logo.png`} width={250} height={250} alt="Logo" className="homepage-logo"/>
            <video autoPlay={true} loop={true} muted={true} className={"background-video"}>
                <source src={"/Videos/rain.mp4"} type={'video/mp4'}/>
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
