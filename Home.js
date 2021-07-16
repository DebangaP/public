import React from 'react';

import { View, Text, Image, StyleSheet, TouchableOpacity, FlatList } from 'react-native';

import { COLORS, icons, SIZES, images } from '../constants'

const Home = () => {

    const [dummyData, setDummyData] = React.useState([
        {
            id: 0,
            name: 'Item 1',
            img: images.rice_farm,
            favourite: false
        },
        {
            id: 1,
            name: 'Item 2',
            img: images.veggies,
            favourite: false
        },
        {
            id: 2,
            name: 'Item 3',
            img: images.rice_farm,
            favourite: false
        }
    ])

    function renderNewItems(item, index) {
        return (
            <View style={{ alignItems: 'center', justifyContent: 'center', marginHorizontal: SIZES.base }}>
                <Image 
                    source={item.img}
                    resizeMode="contain"
                    style={{
                        width: SIZES.width * 0.2,
                        height: "90%",
                        borderRadius: 6
                    }}
                />
                <View
                    style={{
                        position: 'absolute',
                        bottom: "20%",
                        right: 0,
                        backgroundColor: COLORS.primary,
                        paddingHorizontal: SIZES.base,
                        borderLeftTopRadius: 10,
                        borderRightTopRadius: 10
                    }}
                >
                    <Text style={{ color: COLORS.white, fontSize: 14 }} > {item.name} </Text>
                </View>
                <TouchableOpacity
                    style={{
                        position: 'absolute',
                        top: '33%',
                        left: 7
                    }}
                    onPress={ () => {console.log(" Favourite clicked")} }
                >
                    <Image 
                            source={item.favourite ? icons.ico_heart_filled : icons.ico_heart }
                            resizeMode="contain"
                            style={{
                                width: 20,
                                height: 20
                            }}
                    />
                    
                </TouchableOpacity>
            </View>
        )
    }

    return (

        // {/* Main Container  */}
        <View style={ styles.container }>

            {/* Top View Container */}
            <View style={{ height: "25%", backgroundColor: COLORS.white}}>
                <View
                    style={{
                        flex: 1,
                        borderBottomLeftRadius: 15,
                        borderBottomRightRadius: 15,
                        backgroundColor: COLORS.bluecornflower
                    }}
                >
                    <View style={{ marginTop: SIZES.padding * 0.5, marginHorizontal: SIZES.padding * 1.5 }}>
                        <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' }}>
                            <Text style={{ color: COLORS.black, fontSize: 20}}> New Items! </Text>
                            <TouchableOpacity
                                onPress={ () => {console.log("touched...")}}
                            >
                                <Image
                                    source={icons.ico_focus}
                                    resizeMode="contain"
                                    style={{
                                        width: 20,
                                        height: 20
                                    }}
                                >
                                </Image>
                            </TouchableOpacity>
                        </View>
                        
                        <View style={{ marginTop: SIZES.base * -4 }}>
                            <FlatList 
                                horizontal
                                showHorizontalScrollIndicator={false}
                                data={dummyData}
                                keyExtractor={ item => item.id.toString()}
                                renderItem={ ({item, index}) => renderNewItems(item, index)}
                            />

                        </View>
                    </View>
                </View>
            </View>

            {/* Main View Container */}
            <View style={{ height: "50%", backgroundColor: COLORS.white }}>
                <View
                    style={{
                        flex: 1,
                        borderBottomLeftRadius: 50,
                        borderBottomRightRadius: 50,
                        backgroundColor: COLORS.white
                    }}
                >
                    <View style={{ marginTop: SIZES.font, marginHorizontal: SIZES.padding}}>
                       
                        <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' }}>
                            <Text style={{ color: COLORS.secondary, fontSize: 20}}> Today's Items </Text>
                            
                            <TouchableOpacity onPress={ () => {console.log("pressed")}}>
                                <Text style={{ color: COLORS.secondary, fontSize: 15}}> See All </Text>
                            </TouchableOpacity>
                        </View>
                    </View>

                    <View style={{ flexDirection: 'row', height: "88%", marginTop: SIZES.base }}>
                        
                        <View style={{ flex: 1, backgroundColor: COLORS.custom }}>
                            <TouchableOpacity
                                style={{ flex: 1}}
                                onPress={ () => {console.log("1. here")}}
                            >
                                <Image 
                                    source={images.veggies}
                                    resizeMode="cover"
                                    style={{
                                        width: '95%',
                                        height: '95%',
                                        borderRadius: 30,
                                        left: 5
                                    }}
                                />
                            </TouchableOpacity>
                            <TouchableOpacity
                                style={{ flex: 1}}
                                onPress={ () => {console.log("2. here")}}
                            >
                                <Image 
                                    source={images.rice_farm}
                                    resizeMode="cover"
                                    style={{
                                        width: '95%',
                                        height: '95%',
                                        borderRadius: 30,
                                        left: 5
                                    }}
                                />
                            </TouchableOpacity>
                        </View>
                        
                        <View style={{ flex: 1.3, backgroundColor: COLORS.lightgray}}>
                        <TouchableOpacity
                                style={{ flex: 1}}
                                onPress={ () => {console.log("3. here")}}
                            >
                                <Image 
                                    source={images.rice_farm}
                                    resizeMode="cover"
                                    style={{
                                        width: '95%',
                                        height: '95%',
                                        borderRadius: 30,
                                        left: 5,
                                        top: 5

                                    }}
                                />
                            </TouchableOpacity>
                        </View>
                    </View>

                </View>
            </View>

            {/* Bottom View Container */}
            <View style={{ height: "25%", backgroundColor: COLORS.white }}>
                
            </View>
        </View>
    )
}

const styles = StyleSheet.create({
    container: {
        flex: 1
    }
})

export default Home;
